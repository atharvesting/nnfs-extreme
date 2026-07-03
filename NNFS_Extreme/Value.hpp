// Guide source: https://github.com/roiamiel1/Build-Deep-Learning-From-Scratch/tree/main/stages/stage_01_scalar_values
#pragma once
#include <utility> // pair, make_pair
#include <string> // string
#include <memory> // shared_ptr, make_shared
#include <iostream> // cout, endl
#include <functional> // function
#include <cmath> // pow

class Value;

using ValuePair = std::pair<std::shared_ptr<Value>, std::shared_ptr<Value>>;

class Value : public std::enable_shared_from_this<Value> {
private:
	ValuePair _prev;
	char _op;
	std::function<void()> _backward = nullptr;

	const std::shared_ptr<Value> _make(float data, ValuePair _children, char _op) const {
		return std::make_shared<Value>(data, _children, _op);
	}

public:
	float data;
	float grad = 0.0f;

	Value(float data, ValuePair _children, char _op)
		: data(data), _prev(_children), _op(_op) {
	}

	Value(float data) : data(data), _prev{ nullptr, nullptr }, _op('\0') {}

	void info() {
		std::cout << "Value(" << data << ")" << std::endl;
	}

	/*
	* The following block of code defines a macro that generates a bunch of boilerplate code for operator overloads.
	* This approach eliminates all verbosity that comes with individual definitions for each operator and each
	*	parameter combination.
	* friend functions are used to keep the operator overloads as part of the class and to suppress the implicit
	*	this pointer left-hand-side injection in operator overloads.
	*/
#define DEFINE_MIXED_OPERATOR(op_sym, op_char) \
	friend std::shared_ptr<Value> operator op_sym(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) { \
		return std::make_shared<Value>(lhs->data op_sym rhs->data, std::make_pair(lhs, rhs), op_char); \
	} \
	friend std::shared_ptr<Value> operator op_sym(const std::shared_ptr<Value>& lhs, const float& rhs) { \
		auto r_ptr = std::make_shared<Value>(rhs); \
		return std::make_shared<Value>(lhs->data op_sym r_ptr->data, std::make_pair(lhs, r_ptr), op_char); \
	} \
	friend std::shared_ptr<Value> operator op_sym(const float& lhs, const std::shared_ptr<Value>& rhs) { \
		auto l_ptr = std::make_shared<Value>(lhs); \
		return std::make_shared<Value>(l_ptr->data op_sym rhs->data, std::make_pair(l_ptr, rhs), op_char); \
	}
	DEFINE_MIXED_OPERATOR(+, '+')
		DEFINE_MIXED_OPERATOR(-, '-')
		DEFINE_MIXED_OPERATOR(*, '*')
		DEFINE_MIXED_OPERATOR(/ , '/')

#undef DEFINE_MIXED_OPERATOR

		std::shared_ptr<Value> pow(const float& exponent) {
		auto expo_ptr = std::make_shared<Value>(exponent);
		return std::make_shared<Value>(
			std::pow(this->data, exponent),
			std::make_pair(shared_from_this(), expo_ptr),
			'^'
		);
	}

	std::shared_ptr<Value> operator-() {
		return shared_from_this() * -1.0f;
	}
};

std::shared_ptr<Value> newValue(const float& val);