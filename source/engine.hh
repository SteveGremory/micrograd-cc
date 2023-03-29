#pragma once

#include <algorithm>
#include <array>
#include <iostream>
#include <map>

class Value {
public:
    enum Operations { ADD, SUB, MUL, TANH, NONE };
    static inline std::map<Operations, std::string_view> OpMap{
        {ADD, " + "}, {SUB, " - "}, {MUL, " * "}, {TANH, " th "}};

    double data, grad;
    std::string label;
    std::array<Value*, 2> children;
    Operations operation;

    Value(double data, std::string label);
    Value(double data, Operations operation, std::string label);
    Value(double data, Operations operation, std::array<Value*, 2> children,
          std::string label);
    Value(const Value& val);
    ~Value();

    void backward();

    Value operator+(Value& value);
    Value operator-(Value& value);
    Value operator*(Value& value);
    Value tanh();
    Value& operator=(const Value& value);

    Value tanh(Value& value);

private:
    int add_backward();
    int mul_backward();
    int tanh_backward();
};