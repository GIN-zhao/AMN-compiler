
#include <iostream>
#include <iterator>
#include <string>
template <typename Device> class Base {
public:
  void play() { static_cast<Device *>(this)->print(); }
};

class A : public Base<A> {
private:
  std::string name;

public:
  A() : name("A") {}
  void print() { std::cout << "my name is " << this->name << std::endl; }
};

class B : public Base<B> {
private:
  std::string name;

public:
  B() : name("B") {}
  void print() { std::cout << "my name is " << this->name << std::endl; }
};
int main()

{
  A a;
  a.play();

  B b;
  b.play();
  std::cout << "hello world\n";
}