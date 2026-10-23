/**
 * Weighted best-of-three solver: like best_three_solver, but every valid word
 * is a possible answer, weighted by an empirical prior (scripts/empirical_prior.py
 * --export), so off-list answers and repeats are handled natively.
 *
 * Objective, lexicographic with tolerance --tol:
 *   1. maximize P(solve within 3)
 *   2. maximize P(solve within 2)
 *   3. minimize expected guesses
 * After guesses 1 and 2 the answers split into classes; guess 3 takes the most
 * likely word of its class, so  P(<=3) = sum over classes of max weight.
 * Turns 4-6 then maximize P(solve by turn 6), then expected guesses: turn 4
 * maximizes P(win within 3 more) over a shortlist, turns 5-6 repeat the turn 2-3
 * logic. Expected guesses are estimated as
 *   cost(g) = 1 + sum_{hint groups b != all-green} P(b) * (1 + 0.6 log2(1 + H_b))
 * (H_b = entropy in bits of the group), the same scorer as the Python backtest;
 * it alone picks guesses from turn 7 on.
 *
 * The opener search evaluates first guesses in order of an upper bound on P(<=3)
 * and stops when the bound falls below the best found (exact), or at --max-first.
 *
 * Usage:
 *   weighted_three_solver [--prior FILE] [--top N] [--max-first K]   search opener
 *       (--candidates F also evaluates the openers listed in F; --save-top F writes the best 30)
 *   weighted_three_solver [--prior FILE] --play [first_guess]        interactive guide
 *       (--solved-file F writes the winning word to F; used by ./daily)
 *   weighted_three_solver [--prior FILE] --eval [first_guess]        turn distribution
 *   weighted_three_solver --backtest DAYS.csv PRIORS.bin             walk-forward replay
 * --threads N caps worker threads (default: all cores).
 * Default prior: data/prior_next.csv (word,prob). PRIORS.bin is float32 [days x words]
 * in valid_words.txt order; DAYS.csv has "date,answer" lines; the opener is
 * re-searched whenever the month changes.
 */

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
using namespace std;

typedef uint8_t u8;

static const int NPAT = 243;
static const u8 ALL_GREEN = 242;
static const int SCORE_CAP = 2000;  // answers used when scoring turn-4+ guesses

static double TOL = 1e-6;
static int MAX_FIRST = 300;
static int N_THREADS = 0;  // 0 = hardware_concurrency

static vector<string> load_words(const string& path) {
    ifstream in(path);
    vector<string> out;
    string w;
    while (in >> w) {
        for (char& c : w) c = tolower(c);
        if (w.size() == 5) out.push_back(w);
    }
    return out;
}

static u8 pattern(const string& g, const string& a) {
    int res[5] = {0}, cnt[26] = {0};
    for (int i = 0; i < 5; i++) {
        if (g[i] == a[i]) res[i] = 2;
        else cnt[a[i] - 'a']++;
    }
    for (int i = 0; i < 5; i++)
        if (res[i] == 0 && cnt[g[i] - 'a'] > 0) {
            res[i] = 1;
            cnt[g[i] - 'a']--;
        }
    int p = 0;
    for (int i = 0; i < 5; i++) p = p * 3 + res[i];
    return (u8)p;
}

static string pattern_str(int p) {
    string s(5, '.');
    for (int i = 4; i >= 0; i--, p /= 3) s[i] = "BYG"[p % 3];
    return s;
}

static int parse_pattern(const string& t) {
    if (t.size() != 5) return -1;
    int p = 0;
    for (char ch : t) {
        ch = toupper(ch);
        int d = ch == 'B' || ch == '.' || ch == '0' ? 0 : ch == 'Y' || ch == '1' ? 1 : ch == 'G' || ch == '2' ? 2 : -1;
        if (d < 0) return -1;
        p = p * 3 + d;
    }
    return p;
}

static inline double est_more(double H) { return 1.0 + 0.6 * log2(1.0 + H); }

struct Key {
    double p3 = -1, p2 = 0, ev = 1e18;
};

static bool better(const Key& a, const Key& b) {
    if (a.p3 > b.p3 + TOL) return true;
    if (a.p3 < b.p3 - TOL) return false;
    if (a.p2 > b.p2 + TOL) return true;
    if (a.p2 < b.p2 - TOL) return false;
    return a.ev < b.ev - 1e-12;
}

struct Ctx {
    int N;
    vector<string> words;
    vector<u8> pat;  // pat[g * N + a]
    inline u8 P(int g, int a) const { return pat[(size_t)g * N + a]; }
};

static void build_patterns(Ctx& c) {
    c.pat.resize((size_t)c.N * c.N);
    int nt = N_THREADS > 0 ? N_THREADS : max(1u, thread::hardware_concurrency());
    atomic<int> next(0);
    vector<thread> pool;
    for (int t = 0; t < nt; t++)
        pool.emplace_back([&]() {
            for (int g; (g = next.fetch_add(1)) < c.N;)
                for (int a = 0; a < c.N; a++) c.pat[(size_t)g * c.N + a] = pattern(c.words[g], c.words[a]);
        });
    for (auto& t : pool) t.join();
}

struct Scratch {
    vector<uint32_t> seen;
    vector<double> sw, swl, mx;
    uint32_t gen = 0;
    vector<char> mark;
    Scratch(int N) : seen(NPAT, 0), sw(NPAT), swl(NPAT), mx(NPAT), mark(N, 0) {}
};

static inline double xlog2x(double x) { return x > 0 ? x * log2(x) : 0.0; }

// Score guess g2 on group members (sorted by weight desc). Returns false if it
// provably cannot beat `best` on P(<=3).
static bool score_group_guess(const Ctx& c, Scratch& s, int g2, const vector<int>& mem, const vector<double>& w,
                              const vector<double>& suffix, const Key& best, Key& out) {
    const u8* row = &c.pat[(size_t)g2 * c.N];
    s.gen++;
    double p3 = 0;
    int n = mem.size();
    for (int i = 0; i < n; i++) {
        int a = mem[i];
        u8 p = row[a];
        double wa = w[a];
        if (s.seen[p] != s.gen) {
            s.seen[p] = s.gen;
            s.sw[p] = wa;
            s.swl[p] = xlog2x(wa);
            s.mx[p] = wa;
            p3 += wa;  // members sorted desc, so first in class is its max
        } else {
            s.sw[p] += wa;
            s.swl[p] += xlog2x(wa);
        }
        if (p3 + suffix[i + 1] < best.p3 - TOL) return false;
    }
    double p2 = 0, ev = 0;
    for (int i = 0; i < n; i++) {
        u8 p = row[mem[i]];
        if (s.seen[p] != s.gen) continue;
        s.seen[p] = s.gen - 1;  // visit each class once
        if (p == ALL_GREEN) {
            p2 += s.sw[p];
            ev += 2 * s.sw[p];
            continue;
        }
        double m = s.mx[p], R = s.sw[p] - m;
        ev += 3 * m;
        if (R > 1e-300) {
            double H = log2(R) - (s.swl[p] - xlog2x(m)) / R;
            ev += R * (3 + est_more(max(H, 0.0)));
        }
    }
    out = {p3, p2, ev};
    return true;
}

// Best second guess for a group of answers. mem must be sorted by weight desc.
static pair<int, Key> solve_group(const Ctx& c, Scratch& s, const vector<int>& mem, const vector<double>& w,
                                  const vector<int>& order) {
    int n = mem.size();
    if (n == 1) return {mem[0], {w[mem[0]], w[mem[0]], 2 * w[mem[0]]}};
    vector<double> suffix(n + 1, 0);
    for (int i = n - 1; i >= 0; i--) suffix[i] = suffix[i + 1] + w[mem[i]];
    Key best;
    int bg = -1;
    Key k;
    for (int g : mem)
        if (score_group_guess(c, s, g, mem, w, suffix, best, k) && better(k, best)) best = k, bg = g;
    for (int g : mem) s.mark[g] = 1;
    for (int g : order) {
        if (s.mark[g]) continue;
        if (score_group_guess(c, s, g, mem, w, suffix, best, k) && better(k, best)) best = k, bg = g;
    }
    for (int g : mem) s.mark[g] = 0;
    return {bg, best};
}

// Estimated expected guesses for each guess tried (all guesses, or `restrict_to`).
static vector<pair<double, int>> greedy_costs(const Ctx& c, Scratch& s, vector<int> cand, const vector<double>& w,
                                              const vector<int>* restrict_to = nullptr) {
    sort(cand.begin(), cand.end(), [&](int x, int y) { return w[x] > w[y]; });
    if ((int)cand.size() > SCORE_CAP) cand.resize(SCORE_CAP);
    double W = 0;
    for (int a : cand) W += w[a];
    vector<pair<double, int>> out;
    auto try_guess = [&](int g) {
        const u8* row = &c.pat[(size_t)g * c.N];
        s.gen++;
        for (int a : cand) {
            u8 p = row[a];
            double q = w[a] / W;
            if (s.seen[p] != s.gen) s.seen[p] = s.gen, s.sw[p] = 0, s.swl[p] = 0;
            s.sw[p] += q;
            s.swl[p] += xlog2x(q);
        }
        double cost = 1;
        for (int a : cand) {
            u8 p = row[a];
            if (s.seen[p] != s.gen) continue;
            s.seen[p] = s.gen - 1;
            if (p == ALL_GREEN) continue;
            double H = log2(s.sw[p]) - s.swl[p] / s.sw[p];
            cost += s.sw[p] * est_more(max(H, 0.0));
        }
        out.push_back({cost, g});
    };
    if (restrict_to) for (int g : *restrict_to) try_guess(g);
    else for (int g = 0; g < c.N; g++) try_guess(g);
    return out;
}

// Minimize estimated expected guesses.
static int greedy_guess(const Ctx& c, Scratch& s, const vector<int>& cand, const vector<double>& w,
                        const vector<int>* restrict_to = nullptr) {
    if (cand.size() == 1) return cand[0];
    auto costs = greedy_costs(c, s, cand, w, restrict_to);
    return min_element(costs.begin(), costs.end())->second;
}

// Most likely remaining word; ties broken by expected guesses.
static int third_guess(const Ctx& c, Scratch& s, const vector<int>& cand, const vector<double>& w) {
    double m = 0;
    for (int a : cand) m = max(m, w[a]);
    vector<int> top;
    for (int a : cand) if (w[a] >= m * (1 - 1e-9)) top.push_back(a);
    if (top.size() == 1) return top[0];
    return greedy_guess(c, s, cand, w, &top);
}

static vector<int> sorted_by_weight(const vector<int>& v, const vector<double>& w) {
    vector<int> r = v;
    stable_sort(r.begin(), r.end(), [&](int x, int y) { return w[x] > w[y]; });
    return r;
}

// With three guesses left: maximize P(solve within those three), then expected
// guesses. Exact over a shortlist (best by expected guesses + most likely words).
static const int SAFE_SHORTLIST = 40;
static int safe3_guess(const Ctx& c, Scratch& s, const vector<int>& cand, const vector<double>& w,
                       const vector<int>& order) {
    auto costs = greedy_costs(c, s, cand, w);
    sort(costs.begin(), costs.end());
    vector<int> shortlist;
    vector<double> ev(c.N, 0);
    for (auto& [cost, g] : costs) ev[g] = cost;
    for (int i = 0; i < (int)costs.size() && i < SAFE_SHORTLIST; i++) shortlist.push_back(costs[i].second);
    vector<int> sc = sorted_by_weight(cand, w);
    for (int i = 0; i < (int)sc.size() && i < SAFE_SHORTLIST / 2; i++) shortlist.push_back(sc[i]);
    int best = shortlist[0];
    double best_p = -1;
    vector<vector<int>> groups(NPAT);
    for (int g : shortlist) {
        for (auto& v : groups) v.clear();
        for (int a : sc) groups[c.P(g, a)].push_back(a);  // stays weight-sorted
        double p = 0;
        for (int q = 0; q < NPAT; q++) {
            if (groups[q].empty()) continue;
            if (q == ALL_GREEN) p += w[groups[q][0]];
            else p += solve_group(c, s, groups[q], w, order).second.p3;
        }
        if (p > best_p + TOL || (p > best_p - TOL && ev[g] < ev[best])) best_p = p, best = g;
    }
    return best;
}

// Guess for `turn` (2..). Turns 2-3 and 5-6 reuse the "win within two" logic:
// 2 and 5 maximize sum over hint classes of the class's top weight, 3 and 6 take
// the top word; 4 maximizes P(win within 3 more); 7+ minimizes expected guesses.
static int next_guess(const Ctx& c, Scratch& s, const vector<int>& cand, const vector<double>& w_abs,
                      const vector<int>& order, int turn) {
    if (cand.size() == 1) return cand[0];
    double W = 0;
    for (int a : cand) W += w_abs[a];
    vector<double> w(w_abs.size(), 0);  // renormalize so tolerances are relative to what is left
    for (int a : cand) w[a] = w_abs[a] / W;
    if (turn == 2 || turn == 5) return solve_group(c, s, sorted_by_weight(cand, w), w, order).first;
    if (turn == 3 || turn == 6) return third_guess(c, s, cand, w);
    if (turn == 4) return safe3_guess(c, s, cand, w, order);
    return greedy_guess(c, s, cand, w);
}

struct FirstResult {
    int guess = -1;
    double bound = 0;
    Key key;
};

// Opener search. Evaluates `extra` (e.g. yesterday's best openers) plus the top
// MAX_FIRST by bound. Returns results sorted best first. The first queued opener
// (`front` if it is queued, e.g. last month's winner) is evaluated alone first so
// the others can be dropped early against its P3; this only changes the order.
static vector<FirstResult> search_first(const Ctx& c, const vector<double>& w, bool verbose,
                                        const vector<int>& extra = {}, int front = -1) {
    int N = c.N;
    vector<int> byw(N);
    iota(byw.begin(), byw.end(), 0);
    stable_sort(byw.begin(), byw.end(), [&](int x, int y) { return w[x] > w[y]; });

    // Upper bound on P(<=3) for each first guess: per hint group, the top 243 weights.
    vector<FirstResult> cands(N);
    {
        int nt = N_THREADS > 0 ? N_THREADS : max(1u, thread::hardware_concurrency());
        atomic<int> next(0);
        vector<thread> pool;
        for (int t = 0; t < nt; t++)
            pool.emplace_back([&]() {
                vector<int> cnt(NPAT);
                for (int g; (g = next.fetch_add(1)) < N;) {
                    fill(cnt.begin(), cnt.end(), 0);
                    double b = 0;
                    const u8* row = &c.pat[(size_t)g * N];
                    for (int a : byw) {
                        u8 p = row[a];
                        if (cnt[p]++ < (p == ALL_GREEN ? 1 : NPAT)) b += w[a];
                    }
                    cands[g] = {g, b, Key()};
                }
            });
        for (auto& t : pool) t.join();
    }
    vector<FirstResult> by_guess = cands;
    sort(cands.begin(), cands.end(), [](const FirstResult& x, const FirstResult& y) { return x.bound > y.bound; });
    vector<FirstResult> todo;
    vector<char> queued(N, 0);
    for (int g : extra)
        if (!queued[g]) queued[g] = 1, todo.push_back(by_guess[g]);
    for (int i = 0; i < N && i < MAX_FIRST; i++)
        if (!queued[cands[i].guess]) queued[cands[i].guess] = 1, todo.push_back(cands[i]);
    bool capped = MAX_FIRST < N && cands[min(MAX_FIRST, N - 1)].bound >= 0;
    for (int i = 1; i < (int)todo.size(); i++)
        if (todo[i].guess == front) { rotate(todo.begin(), todo.begin() + i, todo.begin() + i + 1); break; }

    vector<int> order = byw;  // guess order for group solving: likely answers first
    int nt = N_THREADS > 0 ? N_THREADS : max(1u, thread::hardware_concurrency());
    atomic<int> next(0), done(0);
    mutex mu;
    double best_p3 = 0;
    int pruned_count = 0;
    vector<FirstResult> evaluated;
    auto work = [&](int limit) {
        Scratch s(N);
        vector<vector<int>> groups(NPAT);
        for (;;) {
            int i = next.fetch_add(1);
            if (i >= limit) break;
            FirstResult r = todo[i];
            {
                lock_guard<mutex> lk(mu);
                if (r.bound < best_p3 - TOL) continue;  // cannot win
            }
            for (auto& v : groups) v.clear();
            const u8* row = &c.pat[(size_t)r.guess * N];
            for (int a : byw) groups[row[a]].push_back(a);  // already weight-sorted
            // Per-group upper bound on P(<=3) (top NPAT weights, as in r.bound). Solve the
            // loosest groups first and drop the opener once solved + remaining bounds
            // cannot reach the best P3 found so far (exact: never drops a winner).
            vector<pair<double, int>> todo_groups;  // (bound, pattern); singletons are exact
            double ub = 0;
            for (int p = 0; p < NPAT; p++) {
                const auto& g = groups[p];
                if (g.empty()) continue;
                double b = 0;
                for (int j = 0; j < (int)g.size() && j < (p == ALL_GREEN ? 1 : NPAT); j++) b += w[g[j]];
                ub += b;
                todo_groups.push_back({b, p});
            }
            sort(todo_groups.begin(), todo_groups.end(), greater<>());
            Key k{0, 0, 0};
            bool pruned = false;
            for (auto& [b, p] : todo_groups) {
                if (p == ALL_GREEN) {
                    double wg = w[groups[p][0]];
                    k.p3 += wg, k.p2 += wg, k.ev += wg;
                    continue;
                }
                Key gk = solve_group(c, s, groups[p], w, order).second;
                k.p3 += gk.p3, k.p2 += gk.p2, k.ev += gk.ev;
                ub -= b - gk.p3;
                double cur_best;
                {
                    lock_guard<mutex> lk(mu);
                    cur_best = best_p3;
                }
                if (ub < cur_best - TOL) { pruned = true; break; }
            }
            if (pruned) {
                lock_guard<mutex> lk(mu);
                ++pruned_count;
                continue;
            }
            r.key = k;
            lock_guard<mutex> lk(mu);
            evaluated.push_back(r);
            best_p3 = max(best_p3, k.p3);
            int d = ++done;
            if (verbose && d % 10 == 0)
                cerr << "\r  evaluated " << d << " openers, best P3 " << 100 * best_p3 << "%" << flush;
        }
    };
    work(1);  // seed best_p3
    next = 1;
    vector<thread> pool;
    for (int t = 0; t < nt; t++) pool.emplace_back(work, (int)todo.size());
    for (auto& t : pool) t.join();
    if (verbose) cerr << "\n";
    if (verbose || getenv("W3_STATS"))
        cerr << "  openers: " << todo.size() << " queued, " << evaluated.size() << " fully evaluated, "
             << pruned_count << " dropped early\n";
    sort(evaluated.begin(), evaluated.end(), [](const FirstResult& x, const FirstResult& y) { return better(x.key, y.key); });
    if (verbose && capped)
        cerr << "  note: stopped at --max-first " << MAX_FIRST << "; result is best among those (not proven optimal)\n";
    return evaluated;
}

// Play one game with the strategy; returns turns used (capped at 20).
static int play_game(const Ctx& c, Scratch& s, const vector<double>& w, const vector<int>& order,
                     int first, int answer) {
    vector<int> cand;
    for (int a = 0; a < c.N; a++) if (w[a] > 0) cand.push_back(a);
    int g = first;
    for (int turn = 1; turn <= 20; turn++) {
        u8 fb = c.P(g, answer);
        if (fb == ALL_GREEN) return turn;
        vector<int> nc;
        for (int a : cand) if (c.P(g, a) == fb) nc.push_back(a);
        cand.swap(nc);
        if (cand.empty()) return 99;  // answer had zero prior weight
        g = next_guess(c, s, cand, w, order, turn + 1);
    }
    return 21;
}

static vector<double> load_prior_csv(const string& path, const unordered_map<string, int>& gidx, int N) {
    ifstream in(path);
    if (!in) { cerr << "cannot open prior file " << path << " (run scripts/empirical_prior.py --export)\n"; exit(1); }
    vector<double> w(N, 0);
    string line;
    getline(in, line);
    while (getline(in, line)) {
        size_t p = line.find(',');
        if (p == string::npos) continue;
        string word = line.substr(0, p);
        for (char& ch : word) ch = tolower(ch);
        auto it = gidx.find(word);
        if (it != gidx.end()) w[it->second] = atof(line.c_str() + p + 1);
    }
    double t = accumulate(w.begin(), w.end(), 0.0);
    for (double& x : w) x /= t;
    return w;
}

static vector<int> weight_order(const vector<double>& w) {
    vector<int> o(w.size());
    iota(o.begin(), o.end(), 0);
    stable_sort(o.begin(), o.end(), [&](int x, int y) { return w[x] > w[y]; });
    return o;
}

static void interactive(const Ctx& c, const vector<double>& w, const unordered_map<string, int>& gidx, int first,
                        const string& solved_file) {
    Scratch s(c.N);
    vector<int> order = weight_order(w);
    printf("Enter hints as 5 letters: B/. gray, Y yellow, G green (e.g. BYBBG).\n"
           "Played something else? Enter \"word HINTS\". \"q\" quits, \"r\" restarts.\n");
    vector<int> all;
    for (int a = 0; a < c.N; a++) if (w[a] > 0) all.push_back(a);
    vector<int> cand = all;
    int turn = 1, g = first;
    for (;;) {
        double W = 0;
        for (int a : cand) W += w[a];
        printf("\nTurn %d: %zu possible. Play: %s", turn, cand.size(), c.words[g].c_str());
        bool g_possible = find(cand.begin(), cand.end(), g) != cand.end();
        if (g_possible) printf("  (%.1f%% it is the answer)", 100 * w[g] / W);
        else printf("  (not a possible answer; splits the rest)");
        printf("\n  most likely:");
        vector<int> top = sorted_by_weight(cand, w);
        for (int i = 0; i < (int)top.size() && i < 10; i++)
            printf(" %s %.1f%%", c.words[top[i]].c_str(), 100 * w[top[i]] / W);
        printf("\n> ");
        fflush(stdout);
        string line;
        if (!getline(cin, line) || line == "q") return;
        if (line == "r") { cand = all, turn = 1, g = first; continue; }
        string word, h;
        size_t sp = line.find(' ');
        if (sp == string::npos) h = line;
        else word = line.substr(0, sp), h = line.substr(sp + 1);
        for (char& ch : word) ch = tolower(ch);
        int played = g;
        if (!word.empty()) {
            auto it = gidx.find(word);
            if (it == gidx.end()) { printf("  not a valid guess: %s\n", word.c_str()); continue; }
            played = it->second;
        }
        int p = parse_pattern(h);
        if (p < 0) { printf("  bad hints: %s\n", h.c_str()); continue; }
        if (p == ALL_GREEN) {
            printf("  Solved in %d!\n", turn);
            if (!solved_file.empty()) ofstream(solved_file) << c.words[played] << "\n";
            return;
        }
        vector<int> nc;
        for (int a : cand) if (c.P(played, a) == p) nc.push_back(a);
        if (nc.empty()) { printf("  no valid word fits; check the hints\n"); continue; }
        cand.swap(nc);
        turn++;
        g = next_guess(c, s, cand, w, order, turn);
    }
}

static void evaluate(const Ctx& c, const vector<double>& w, int first) {
    Scratch s(c.N);
    vector<int> order = weight_order(w);
    vector<double> hist(22, 0);
    double ev = 0;
    for (int a = 0; a < c.N; a++) {
        if (w[a] <= 0) continue;
        int t = play_game(c, s, w, order, first, a);
        hist[min(t, 21)] += w[a];
        ev += w[a] * t;
    }
    printf("Opening %s, prior-weighted over %d words: expected guesses %.4f\n", c.words[first].c_str(), c.N, ev);
    double cum = 0;
    for (int t = 1; t < 22; t++) {
        if (hist[t] <= 0) continue;
        cum += hist[t];
        printf("turn %2d: %7.3f%%  cumulative %7.3f%%\n", t, 100 * hist[t], 100 * cum);
    }
}

static void backtest(const Ctx& c, const string& days_path, const string& bin_path) {
    ifstream dn(days_path);
    vector<pair<string, string>> days;
    string line;
    while (getline(dn, line)) {
        size_t p = line.find(',');
        if (p == string::npos || line.substr(0, 4) == "date") continue;
        string a = line.substr(p + 1);
        while (!a.empty() && (a.back() == '\r' || a.back() == ' ')) a.pop_back();
        for (char& ch : a) ch = tolower(ch);
        days.push_back({line.substr(0, p), a});
    }
    unordered_map<string, int> gidx;
    for (int i = 0; i < c.N; i++) gidx[c.words[i]] = i;
    FILE* f = fopen(bin_path.c_str(), "rb");
    if (!f) { cerr << "cannot open " << bin_path << "\n"; exit(1); }
    vector<float> buf(c.N);
    vector<double> w(c.N);
    Scratch s(c.N);
    string month;
    int first = -1;  // last month's opener goes first in the next search
    vector<double> searched_w;  // prior the current opener was searched on
    for (auto& [date, ans] : days) {
        if (fread(buf.data(), sizeof(float), c.N, f) != (size_t)c.N) { cerr << "short priors file\n"; exit(1); }
        double t = 0;
        for (int i = 0; i < c.N; i++) t += (w[i] = buf[i]);
        for (double& x : w) x /= t;
        vector<int> order = weight_order(w);
        if (date.substr(0, 7) != month && w != searched_w) {  // same prior (e.g. orig-list): same opener
            month = date.substr(0, 7);
            searched_w = w;
            auto res = search_first(c, w, false, {}, first);
            first = res[0].guess;
            cerr << "  " << month << " opener " << c.words[first] << " P3 " << 100 * res[0].key.p3 << "%\n";
        }
        auto it = gidx.find(ans);
        if (it == gidx.end()) { cerr << "unknown answer " << ans << "\n"; exit(1); }
        int turns = play_game(c, s, w, order, first, it->second);
        printf("%s,%s,%d,%s\n", date.c_str(), ans.c_str(), turns, c.words[first].c_str());
        fflush(stdout);
    }
    fclose(f);
}

int main(int argc, char** argv) {
    string prior_path = "data/prior_next.csv", mode = "search", first_arg, days_path, bin_path, solved_file;
    string cand_path, save_top_path;
    int top_n = 20;
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "--prior" && i + 1 < argc) prior_path = argv[++i];
        else if (a == "--top" && i + 1 < argc) top_n = atoi(argv[++i]);
        else if (a == "--max-first" && i + 1 < argc) MAX_FIRST = atoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) N_THREADS = atoi(argv[++i]);
        else if (a == "--tol" && i + 1 < argc) TOL = atof(argv[++i]);
        else if (a == "--solved-file" && i + 1 < argc) solved_file = argv[++i];
        else if (a == "--candidates" && i + 1 < argc) cand_path = argv[++i];
        else if (a == "--save-top" && i + 1 < argc) save_top_path = argv[++i];
        else if (a == "--play" || a == "--eval") {
            mode = a.substr(2);
            if (i + 1 < argc && argv[i + 1][0] != '-') first_arg = argv[++i];
        } else if (a == "--backtest" && i + 2 < argc) mode = "backtest", days_path = argv[++i], bin_path = argv[++i];
        else { cerr << "unknown argument " << a << "\n"; return 1; }
    }

    Ctx c;
    c.words = load_words("valid_words.txt");
    c.N = c.words.size();
    unordered_map<string, int> gidx;
    for (int i = 0; i < c.N; i++) gidx[c.words[i]] = i;
    cerr << c.N << " words; building pattern table...\n";
    build_patterns(c);

    if (mode == "backtest") {
        backtest(c, days_path, bin_path);
        return 0;
    }
    vector<double> w = load_prior_csv(prior_path, gidx, c.N);

    int first = -1;
    if (!first_arg.empty()) {
        for (char& ch : first_arg) ch = tolower(ch);
        if (!gidx.count(first_arg)) { cerr << "not a valid guess: " << first_arg << "\n"; return 1; }
        first = gidx[first_arg];
    }
    if (first < 0 || mode == "search") {
        vector<int> extra;
        if (!cand_path.empty())
            for (auto& x : load_words(cand_path))
                if (gidx.count(x)) extra.push_back(gidx[x]);
        auto res = search_first(c, w, true, extra);
        if (!save_top_path.empty()) {
            ofstream out(save_top_path);
            for (int i = 0; i < (int)res.size() && i < 30; i++) out << c.words[res[i].guess] << "\n";
        }
        if (mode == "search") {
            printf("Top first guesses (%zu evaluated; others pruned by bound)\n", res.size());
            printf("%-4s %-6s %8s %8s %8s %8s\n", "#", "guess", "P<=3", "P<=2", "P<=1", "E[g]*");
            for (int i = 0; i < (int)res.size() && i < top_n; i++)
                printf("%-4d %-6s %7.3f%% %7.3f%% %7.3f%% %8.4f\n", i + 1, c.words[res[i].guess].c_str(),
                       100 * res[i].key.p3, 100 * res[i].key.p2, 100 * w[res[i].guess], res[i].key.ev);
            printf("* E[g] uses the entropy estimate for turns after 3; --eval simulates it exactly.\n");

            int g1 = res[0].guess;
            Scratch s(c.N);
            vector<int> order = weight_order(w);
            vector<vector<int>> groups(NPAT);
            for (int a : order) groups[c.P(g1, a)].push_back(a);
            vector<int> ps;
            for (int p = 0; p < NPAT; p++) if (!groups[p].empty()) ps.push_back(p);
            sort(ps.begin(), ps.end(), [&](int x, int y) {
                double wx = 0, wy = 0;
                for (int a : groups[x]) wx += w[a];
                for (int a : groups[y]) wy += w[a];
                return wx > wy;
            });
            printf("\nStrategy after %s (B gray, Y yellow, G green)\n", c.words[g1].c_str());
            printf("%-7s %6s %7s  %-6s %8s  %s\n", "pattern", "words", "P(grp)", "guess2", "P3|grp", "in-group");
            for (int p : ps) {
                double wg = 0;
                for (int a : groups[p]) wg += w[a];
                if (p == ALL_GREEN) { printf("%-7s %6d %6.2f%%  (solved)\n", pattern_str(p).c_str(), 1, 100 * wg); continue; }
                auto [g2, k] = solve_group(c, s, groups[p], w, order);
                bool in = w[g2] > 0 && c.P(g1, g2) == p;
                printf("%-7s %6zu %6.2f%%  %-6s %7.1f%%  %s\n", pattern_str(p).c_str(), groups[p].size(), 100 * wg,
                       c.words[g2].c_str(), 100 * k.p3 / wg, in ? "yes" : "no");
            }
            return 0;
        }
        first = res[0].guess;
    }
    if (mode == "play") interactive(c, w, gidx, first, solved_file);
    else evaluate(c, w, first);
    return 0;
}
