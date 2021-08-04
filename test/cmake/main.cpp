#include "saga/all.hpp"

#include <cassert>

int main() {
  auto r = saga::particles<saga::types::cpu::single_float_precision>(10);
  assert(r.size() == 10);
  return 0;
}
