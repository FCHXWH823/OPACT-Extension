#include <vector>
#include <string>
#include <utility>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <iostream>

/**
 * Extract the first Pareto frontier ("Front 0: …") from a text file.
 *
 * @param path      path to the .txt file
 * @return          vector of (f1 , f2) pairs belonging to Front 0
 *
 * Throws std::runtime_error if the file cannot be opened or
 * if no “Front 0:” line is found.
 */
std::vector<std::pair<double,double>>
readFirstFrontier(const std::string& path)
{
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Cannot open file: " + path);

    std::string line;
    const std::regex pair_re(
        R"(\(\s*([+-]?(?:\d*\.\d+|\d+\.?)(?:[eE][+-]?\d+)?)\s*,\s*([+-]?(?:\d*\.\d+|\d+\.?)(?:[eE][+-]?\d+)?)\s*\))");

    while (std::getline(in, line))
    {
        if (line.rfind("Front 0:", 0) == 0)          // starts with "Front 0:"
        {
            std::vector<std::pair<double,double>> frontier;
            std::sregex_iterator it(line.begin(), line.end(), pair_re);
            std::sregex_iterator end;

            for (; it != end; ++it)
            {
                double x = std::stod((*it)[1]);
                double y = std::stod((*it)[2]);
                frontier.emplace_back(x, y);
            }
            return frontier;                         // done
        }
    }
    throw std::runtime_error("No \"Front 0:\" line found in " + path);
}


int main()
{
    try
    {
        auto front0 = readFirstFrontier("/Users/fch/Python/OPACT-Extension/Training_log_0.0001_ste_counts/non-dominated-sorting.txt");
        std::cout << "Front 0 contains " << front0.size() << " points:\n";
        for (auto [f1, f2] : front0)
            std::cout << '(' << f1 << ", " << f2 << ")\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << '\n';
    }
}