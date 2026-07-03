// Guide source: https://github.com/roiamiel1/Build-Deep-Learning-From-Scratch/tree/main/stages/stage_01_scalar_values
#pragma once
#include <utility> // pair, make_pair
#include <string> // string
#include <memory> // shared_ptr, make_shared
#include <iostream> // cout, endl
#include <functional> // function
#include <cmath> // pow
#include <set> // set

class Value;

using ValuePtr = std::shared_ptr<Value>;
using ValuePair = std::pair<ValuePtr, ValuePtr>;

class Value : public std::enable_shared_from_this<Value> {
public:
	float data;
	float grad = 0.0f;
	ValuePair _prev;
	char _op;
	std::function<void()> _backward = nullptr;
	
	Value(float data, ValuePair _children, char _op)
		: data(data), _prev(_children), _op(_op) {
	}

	Value(float data) : data(data), _prev{ nullptr, nullptr }, _op('\0') {}

	void info() {
		std::cout << "Value(data=" << data << ", op=" << _op << ")" << std::endl;
	}

	const ValuePtr _make(float data, ValuePair _children, char _op) const {
		return std::make_shared<Value>(data, _children, _op);
	}

	/*
	* The following block of code defines a macro that generates a bunch of boilerplate code for operator overloads.
	* This approach eliminates all verbosity that comes with individual definitions for each operator and each
	*	parameter combination.
	* friend functions are used to keep the operator overloads as part of the class and to suppress the implicit
	*	this pointer left-hand-side injection in operator overloads.
	*/
#define DEFINE_MIXED_OPERATOR(op_sym, op_char) \
	friend ValuePtr operator op_sym(const ValuePtr& lhs, const ValuePtr& rhs) { \
		return std::make_shared<Value>(lhs->data op_sym rhs->data, std::make_pair(lhs, rhs), op_char); \
	} \
	friend ValuePtr operator op_sym(const ValuePtr& lhs, const float& rhs) { \
		auto r_ptr = std::make_shared<Value>(rhs); \
		return std::make_shared<Value>(lhs->data op_sym r_ptr->data, std::make_pair(lhs, r_ptr), op_char); \
	} \
	friend ValuePtr operator op_sym(const float& lhs, const ValuePtr& rhs) { \
		auto l_ptr = std::make_shared<Value>(lhs); \
		return std::make_shared<Value>(l_ptr->data op_sym rhs->data, std::make_pair(l_ptr, rhs), op_char); \
	}
	DEFINE_MIXED_OPERATOR(+, '+')
	//DEFINE_MIXED_OPERATOR(-, '-')
	DEFINE_MIXED_OPERATOR(*, '*')
	//DEFINE_MIXED_OPERATOR(/ , '/')
#undef DEFINE_MIXED_OPERATOR

	ValuePtr pow(const float& exponent) {
		auto expo_ptr = std::make_shared<Value>(exponent);
		return std::make_shared<Value>(
			std::pow(this->data, exponent),
			std::make_pair(shared_from_this(), expo_ptr),
			'^'
		);
	}

	ValuePtr operator-() const {
		return shared_from_this() * -1.0f;
	}

	ValuePtr operator-(ValuePtr other) const {
		// Same as this + (-other)
		return shared_from_this() + (other->operator-());
	}

	ValuePtr operator/(ValuePtr other) {
		// Same as this * other**-1
		return shared_from_this() * other->pow(-1);
	}
};

ValuePtr newValue(const float& val);

using Edge = std::pair<ValuePtr, ValuePtr>;
using NodeNedges = std::pair<
	std::set<ValuePtr>, 
	std::set<Edge>
>;

void trace_internal(ValuePtr root,
					std::set<ValuePtr>& visited,
					NodeNedges& result) {
	if (!root || visited.count(root)) 
		return;

	visited.insert(root);
	result.first.insert(root);

	if (root->_prev.first != nullptr) {
		result.second.insert({ root->_prev.first, root });
		trace_internal(root->_prev.first, visited, result);
	}
	if (current->_prev.second != nullptr) {
		result.second.insert({ root->_prev.second, root });
		trace_internal(root->_prev.second, visited, result);
	}
}

NodeNedges trace(ValuePtr root) {
	NodeNedges result;
	std::set<ValuePtr> visited;
	trace_internal(root, visited, result);
	return result;
}