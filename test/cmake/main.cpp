#include "saga/all.hpp"

#include <cassert>

int main() {
  auto r = saga::particles<saga::types::cpu::sf>(10);
  assert(r.size() == 10);
  return 0;
}
