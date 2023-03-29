#include "engine.hh"
#include "nn.hh"
#include <math.h>

#include <algorithm>
#include <iostream>
#include <vector>

void print_children(Value* value) {
    // Print parent value and it's grad
    printf("Parent Value: %lf (%s) Parent Grad: %lf\n", value->data,
           value->label.c_str(), value->grad);

    // Print child value 1 if it exists and it's grad
    if (value->children[0] != nullptr) {
        printf("\tChild value: %lf (%s) Child Grad: %lf ",
               value->children[0]->data, value->children[0]->label.c_str(),
               value->children[0]->grad);
    } else {
        return;
    }
    // Print operation
    printf("(%s)", Value::OpMap[value->operation].data());
    // Print child value 2 if it exists and it's grad
    if (value->children[1] != nullptr) {
        printf(" Child value: %lf (%s)\tChild Grad: %lf\n",
               value->children[1]->data, value->children[1]->label.c_str(),
               value->children[1]->grad);
    } else if (value->operation == Value::TANH) {
        print_children((Value*)value->children[0]);
    } else {
        return;
    }

    if (value->children[1] != nullptr && value->children[0] != nullptr) {
        print_children((Value*)value->children[0]);
        print_children((Value*)value->children[1]);
    }

    return;
}

int main() {
    // Grad -> 3.96
    Value y(0.21, "y");
    Value o(0.154, "o");

    // Grad -> 3.64
    Value x(0.15, "x");
    Value p(0.246, "p");

    // Grad ->
    Value z(0.443, "z");
    auto a = x + p;
    a.label = "a";

    auto b = y + o;
    b.label = "b";

    auto c = a * b;
    c.label = "c";

    auto d = c * z;
    d.label = "d";

    auto e = d.tanh();

    // final grad is always 0.
    e.grad = 1;
    e.backward();
    d.backward();
    c.backward();
    b.backward();
    a.backward();

    print_children(&e);
}
