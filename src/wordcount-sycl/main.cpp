#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>
#include <cstdlib>

int word_count(const std::vector<char> &input);
int word_count_reference(const std::vector<char> &input);

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  // Paragraph from 'The Raven' by Edgar Allan Poe
  // http://en.wikipedia.org/wiki/The_Raven
  const char raw_input[] = "  But the raven, sitting lonely on the placid bust, spoke only,\n"
                           "  That one word, as if his soul in that one word he did outpour.\n"
                           "  Nothing further then he uttered - not a feather then he fluttered -\n"
                           "  Till I scarcely more than muttered `Other friends have flown before -\n"
                           "  On the morrow he will leave me, as my hopes have flown before.'\n"
                           "  Then the bird said, `Nevermore.'\n";

  std::cout << "Text sample:" << std::endl;
  std::cout << raw_input << std::endl;
  
  std::vector<char> input(raw_input, raw_input + sizeof(raw_input));

  // count words
  int wc = word_count_reference(input);
  std::cout << "Host: Text sample contains " << wc << " words" << std::endl;

  wc = word_count(input);
  std::cout << "Device: Text sample contains " << wc << " words" << std::endl;
  
  std::cout << "Test word count with random inputs\n";
  srand(123);
  bool ok = true;
  const char tab[] = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  const int size = strlen(tab);

  // verify
  for (size_t i = 1; i <= 1e8; i = i * 10) {
    std::vector<char> random_input(i);
    for (size_t c = 0; c < i; c++) random_input[c] = tab[rand() % size];
    if (word_count_reference(random_input) != word_count(random_input)) {
      ok = false;
      break;
    }
  }
  std::cout << (ok ? "PASS" : "FAIL") << std::endl;
  if (!ok) exit(1);
      
  // may take a few seconds to initialize
  const size_t len = 1024*1024*256;  
  std::vector<char> random_input(len);
  for (size_t c = 0; c < len; c++) random_input[c] = tab[rand() % size];

  std::cout << "Performance evaluation for random texts of character length " << len << std::endl;
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) word_count(random_input);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << "Average time of word count: " << time * 1e-9f / repeat << " (s)\n";

  return 0;
}
