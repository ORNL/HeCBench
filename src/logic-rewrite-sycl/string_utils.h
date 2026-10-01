#pragma once

#include <vector>
#include <string>
#include <algorithm>

namespace strUtil {

// trim from start (in place)
inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
}

// trim from end (in place)
inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

// trim from both ends (in place)
inline void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}

inline void split(const std::string & s, std::vector<std::string> & v, const std::string & delim) {
    std::string::size_type pos1, pos2;
    pos2 = s.find(delim);
    pos1 = 0;
    v.clear();
    while (std::string::npos != pos2) {
        v.push_back(s.substr(pos1, pos2-pos1));
    
        pos1 = pos2 + delim.size();
        pos2 = s.find(delim, pos1);
    }
    if (pos1 != s.length())
        v.push_back(s.substr(pos1));
}

inline std::string join(const std::vector<std::string> & v, const std::string & delim) {
    std::string s;
    for (int i = 0; i < v.size(); i++) {
        s += v[i];
        if (i != v.size() - 1)
            s += delim;
    }
    return s;
}

inline std::string concat(const std::vector<std::string> & v) {
    std::string res;
    for (const auto & s : v)
        res += s;
    return res;
}

inline bool startsWith(const std::string & str, const std::string & prefix) {
    return (str.rfind(prefix, 0) == 0);
}

inline bool endsWith(const std::string & str, const std::string & suffix) {
    if (suffix.length() > str.length()) 
        return false;
    
    return (str.rfind(suffix) == (str.length() - suffix.length()));
}

template<typename T>
std::string descWithDefault(const std::string & desc, const T & defaultValue) {
    std::ostringstream ss;
    if (std::is_same<T, bool>::value)
        ss << desc << " [default = " << (defaultValue ? "true" : "false") << "]";
    else
        ss << desc << " [default = " << defaultValue << "]";
    return ss.str(); 
}

} // namespace strUtil
