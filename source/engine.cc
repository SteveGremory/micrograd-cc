#include "engine.hh"
#include <math.h>
#include <memory>

// Constructors/Destructors
Value::Value(double data, std::string label) : data(data), label(label) {
    this->children = {nullptr, nullptr};
    this->grad = 0.0;
    this->operation = Operations::NONE;
};

Value::Value(double data, Operations operation, std::string label)
    : data(data), label(label), operation(operation) {
    this->children = {nullptr, nullptr};
    this->grad = 0.0;
}

Value::Value(double data, Operations operation, std::array<Value*, 2> children,
             std::string label)
    : data(data), label(label), operation(operation), children(children) {
    this->grad = 0.0;
}

Value::~Value() {}

Value::Value(const Value& val) { *this = val; };

// Operator overloads
Value& Value::operator=(const Value& value) {
    this->children = value.children;
    this->data = value.data;
    this->operation = value.operation;

    return *this;
}

Value Value::operator+(Value& value) {
    std::array<Value*, 2> children = {this, &value};
    std::string new_label = this->label + " + " + value.label;
    Value result(this->data + value.data, Operations::ADD, children, new_label);

    return result;
}

Value Value::operator-(Value& value) {
    std::array<Value*, 2> children = {this, &value};

    std::string new_label = this->label + " - " + value.label;
    Value result(this->data - value.data, Operations::SUB, children, new_label);

    return result;
}

Value Value::operator*(Value& value) {
    std::array<Value*, 2> children = {this, &value};

    std::string new_label = this->label + " * " + value.label;
    Value result(this->data * value.data, Operations::MUL, children, new_label);

    return result;
}

Value Value::tanh() {
    std::array<Value*, 2> children = {this, nullptr};

    std::string new_label = "tanh";
    auto n = this->data;
    Value result((exp(2 * n) - 1) / (exp(2 * n) + 1), Operations::TANH,
                 children, new_label);

    return result;
}

// Functions
int Value::add_backward() {
    // Both the previous nodes have the same grad value as the current node
    auto current_children = this->children;

    current_children[0]->grad += 1.0 * this->grad;
    current_children[1]->grad += 1.0 * this->grad;

    return 0;
}

int Value::mul_backward() {
    // Previous node x's grad will be the value of
    // y times the grad of the result
    auto current_children = this->children;

    current_children[0]->grad = current_children[1]->data * this->grad;
    current_children[1]->grad = current_children[0]->data * this->grad;

    return 0;
}

int Value::tanh_backward() {
    auto current_children = this->children;
    auto val = (1 - pow(current_children[0]->data, 2)) * this->grad;
    ;
    current_children[0]->grad = val;

    std::cout << "VALUE: " << val << std::endl;
    return 0;
}

void Value::backward() {
    switch (operation) {
    case ADD:
        add_backward();
        break;
    case SUB:
        std::cerr
            << "A backward function hasn't been implemented for this operation."
            << std::endl;
        break;
    case MUL:
        mul_backward();
        break;
    case TANH:
        tanh_backward();
        break;
    default:
        break;
    };
}