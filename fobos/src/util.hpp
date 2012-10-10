#ifndef UTIL_HPP
#define UTIL_HPP

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <boost/algorithm/string.hpp>

typedef std::vector<std::pair<int, float> > fv_t;

int parse_line(const std::string& line, fv_t& fv, int&y ) {
    // empty line cannot be parsed.
    if (line == "") {
        return -1;
    }

    std::istringstream is(line);
    int y_;

    if (!(is >> y_)) {
        std::cerr << "parse error: no label" << std::endl;
        return -1;
    }
    y = y_; // yの取得

    std::string tmp;

    // 要素数分だけ繰り返す
    while (is >> tmp) {
        std::vector<std::string> ret;
        int id;
        float val;

        boost::algorithm::split(ret, tmp, boost::is_any_of(":")); // tmpを":"で分割

        id = atoi(ret[0].c_str()); // idの取得
        std::istringstream(ret[1]) >> val; // valの取得

        fv.push_back(std::make_pair(id, val));
    }

    if (fv.empty()) return -1;

    return 0;
};

#endif
