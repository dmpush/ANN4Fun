#ifndef __CONSOLE_HPP_
#define __CONSOLE_HPP_

#include <ostream>
#include <string>
#include <deque>

class Console {
std::deque<std::string> actions_;
public:

enum colors {
    black   = 30,
    red     = 31,
    green   = 32,
    yellow  = 33,
    blue    = 34,
    magenta = 35,
    aqua    = 36,
    white   = 37
};

    Console() : actions_{} {};

    Console& fgColor(Console::colors color) {
	actions_.push_back("\033["+std::to_string(color)+"m");
	return *this;
    };
    Console& bgColor(Console::colors color) {
	actions_.push_back("\033["+std::to_string(color+10)+"m");
	return *this;
    };
    Console& clear() {
	actions_.push_back("\033[0m");
	return *this;
    };
    Console& bold() {
	actions_.push_back("\033[1m");
	return *this;
    };
    Console& weak() {
	actions_.push_back("\033[2m");
	return *this;
    };
    Console& italic() {
	actions_.push_back("\033[3m");
	return *this;
    };
    Console& underline() {
	actions_.push_back("\033[4m");
	return *this;
    };
    Console& blink() {
	actions_.push_back("\033[5m");
	return *this;
    };
    Console& strikethrough() {
	actions_.push_back("\033[9m");
	return *this;
    };
    template< class CharT, class Traits>
    std::basic_ostream<CharT, Traits>&  escape(std::basic_ostream<CharT, Traits>& os) {
	for(auto a: actions_)
	    os<<a;
	return os;
    };

};

template< class CharT, class Traits>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os, Console& con) {
    return con.escape(os);
};
#endif
