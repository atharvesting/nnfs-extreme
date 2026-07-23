#pragma once
#include <cmath>
#include <string>

std::string float_to_string(float f) {
	f = std::round(f * 100.0F) / 100.0F; // Rounding trick: https://stackoverflow.com/a/14369745
    auto f_string = std::to_string(f);
    auto it = f_string.find('.');
    f_string.replace(it, 1, "p");
    return f_string.substr(0, f_string.size() - 3);
}