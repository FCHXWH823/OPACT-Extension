#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>


using namespace std;
using std::string;
using std::vector;
using namespace std;


int B = 1024;

using Point = std::pair<double,double>;   // (f1 , f2)


std::vector<std::vector<Point>>
nondominatedSort(const std::vector<Point>& objs)
{
    const std::size_t N = objs.size();
    std::vector<std::size_t> perm(N);                  // permutation of indices
    std::iota(perm.begin(), perm.end(), 0);

    /* 1) sort indices by ascending f1, then ascending f2 */
    std::sort(perm.begin(), perm.end(),
              [&objs](std::size_t a, std::size_t b)
              {
                  if (objs[a].first != objs[b].first)
                      return objs[a].first < objs[b].first;
                  return objs[a].second < objs[b].second;
              });

    std::vector<std::vector<Point>> fronts;            // result
    std::vector<double> minF2;                         // best f2 per front

    /* 2) sweep */
    for (std::size_t id : perm)
    {
        const double f2 = objs[id].second;

        /* first front whose current *best* f2 is < f2
           – duplicates (same f1 & f2) join the same front           */
        std::size_t pos = 0;
        while (pos < minF2.size() && minF2[pos] < f2)
            ++pos;

        if (pos == fronts.size()) {                    // new front needed
            fronts.emplace_back();                     // create front
            minF2.push_back(std::numeric_limits<double>::infinity());
        }

        fronts[pos].push_back(objs[id]);               // store the point
        minF2[pos] = std::min(minF2[pos], f2);         // update best f2
    }
    return fronts;
}

int main(int argc, char* argv[])
{
    filesystem::path dir = "/Users/fch/Python/OPACT-Extension/Training_log_0.0001_ste_counts_general"; // Change to your directory
    Point solution_test, solution_ILP_test;
    vector<Point> solutions;

    for (const auto& entry : filesystem::directory_iterator(dir)) {
        if (entry.path().extension() == ".txt" && entry.path().filename().string().find("non-dominated-sorting") == string::npos) {
            string file = entry.path().string();
            ifstream f;
            f.open(file);
            if (!f) {
                cerr << "Error opening file: " << file << endl;
                continue;
            }
            else{
                cout << "Processing file: " << file << endl;
                // get the last line of the file
                Point solution;
                string last_line;
                string line;
                while (getline(f, line))
                    last_line = line;
                f.close();
                // parse the last line
                // get the area & error from  `step     0 | loss=7216073.000 | area=811212.81 err=61386203136.0000 | row-ok=0.000 λ_row=10.00 constraint=25.00 `
                // first split by ' '
                size_t pos = 0;
                string token;
                vector<string> tokens;
                while ((pos = last_line.find(' ')) != string::npos) {
                    token = last_line.substr(0, pos);
                    tokens.push_back(token);
                    last_line.erase(0, pos + 1);
                }
                for(auto s: tokens){
                    if (s.find("area=") != string::npos) {
                        size_t area_pos = s.find("area=") + 5;
                        solution.first = stod(s.substr(area_pos)) / B;
                    }
                    else if (s.find("med=") != string::npos) {
                        size_t err_pos = s.find("med=") + 4;
                        solution.second = stod(s.substr(err_pos)) / B;
                    }
                }

                solutions.push_back(solution);
                cout << "Solution: area = " << solution.first << ", error = " << solution.second << endl;
                    
            }
            
        }
    }


    // non-dominated sorting & store in non-doimated-sorting.txt
    vector<vector<Point>> fronts = nondominatedSort(solutions);
    std::cout << "Pareto fronts:\n";
    ofstream out("/Users/fch/Python/OPACT-Extension/Training_log_0.0001_ste_counts_general/non-dominated-sorting.txt");
    for (std::size_t i = 0; i < fronts.size(); ++i)
    {
        std::cout << "Front " << i << ": ";
        out << "Front " << i << ": ";
        for (const auto& p : fronts[i]){
            std::cout << "(" << p.first << ", " << p.second << ") ";
            out << "(" << p.first << ", " << p.second << ") ";
        }
        std::cout << '\n';
        out << '\n';
    }
    out.close();
    return 0;
}