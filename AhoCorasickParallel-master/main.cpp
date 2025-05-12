#include "timer.h"
#include "utils.h"
#include "ac.h"

#ifdef _WIN32
#include <windows.h>
#include <cstring>
#include <cstdio>

int optind = 1;
int opterr = 1;
int optopt;
char *optarg;

int getopt(int argc, char * const argv[], const char *optstring) {
    static int optpos = 1;
    if (optind >= argc || argv[optind][0] != '-' || argv[optind][1] == '\0') {
        return -1;
    }
    if (strcmp(argv[optind], "--") == 0) {
        optind++;
        return -1;
    }
    char c = argv[optind][optpos];
    const char *optdecl = strchr(optstring, c);
    if (!optdecl) {
        if (opterr) {
            fprintf(stderr, "Unknown option: -%c\n", c);
        }
        optopt = c;
        if (argv[optind][++optpos] == '\0') {
            optind++;
            optpos = 1;
        }
        return '?';
    }
    if (optdecl[1] == ':') {
        if (argv[optind][optpos+1] != '\0') {
            optarg = &argv[optind][optpos+1];
            optind++;
        } else if (optind+1 < argc) {
            optarg = argv[optind+1];
            optind += 2;
        } else {
            if (opterr) {
                fprintf(stderr, "Option -%c requires an argument\n", c);
            }
            optopt = c;
            optpos = 1;
            optind++;
            return '?';
        }
        optpos = 1;
    } else {
        if (argv[optind][++optpos] == '\0') {
            optind++;
            optpos = 1;
        }
        optarg = NULL;
    }
    return c;
}

#else
#include <unistd.h>
#endif

using namespace std;

const int num_cols = 30;
const int num_rows = 5001;
void generate_DFA(int* dfa, string output[], vector<string> input);
void generate_fail_states(int* dfa, string output[], int fail_state[]);
int get_state_as_int(char ch);
char get_state_as_char(int st);

float ms_difference(struct timeval start, struct timeval end) {
    float ms = (end.tv_sec - start.tv_sec) * 1000;
    ms += (end.tv_usec - start.tv_usec) / 1000;
    return ms;
}

void print_usage() {
    printf("usage: ./acp [-l filename] [-p] [-t]\n"
           "              -l load the filename to create dfa\n"
           "              -p run performance tests\n"
           "              -t run test cases\n"
          );
}

std::vector<std::string> load_input_file(const char* filename, int* n, int num_words_to_load) {
    fstream in;
    in.open(filename, fstream::in);

    vector<string> word_list;
    string word;
    int idx = 0;
    while(std::getline(in, word)) {
        word_list.push_back(word);
        idx += 1;
        if(idx > num_words_to_load)
            break;
    }
    *n = idx;
    in.close();
    return word_list;
}

char* load_tweets(const char* filename, int num, int tweet_length) {
    FILE * tweetfile;
    tweetfile = fopen(filename, "r");
    if(tweetfile == NULL) {
        printf("Error: Cannot open file %s\n", filename);
        return NULL;
    }
    char* tweets;
    tweets =  (char *) malloc(num*tweet_length*sizeof(char));
    int idx = 0;
    char ch;
    while(idx < num) {
        if(feof(tweetfile))
            break;
        int i = 0;
        do{
            ch = fgetc(tweetfile);
            tweets[idx*tweet_length + i] = ch;
            i+=1;
        }while(i<tweet_length && ch!=10 && ch!='\n');

        while(i<tweet_length) {
            tweets[idx*tweet_length + i] = ' ';
            i+= 1;
        }
        idx += 1;
    }
    fclose(tweetfile);

    return tweets;
}

void print_dfa(int* dfa, int rows, int cols) {
    const char* format_int = "%4d|";
    const char* format_char = "%4c|";
    const char* format_str = "%4s|";
    printf(format_str, " ");
    printf(format_char, '0');
    for(int i=1; i<cols; i++) {
        printf(format_char, get_state_as_char(i));
    }
    cout<<endl;
    for(int i=0; i<cols+1; i++) {
        printf("%4s", "-----");
    }
    cout<<endl;
    for(int i=0; i<rows; i++) {
        printf(format_int, i);
        printf(format_int, dfa[i*num_cols + 0]);
        for(int j=1; j<cols; j++) {
            if(dfa[i*num_cols + j] != 0)
                printf("%4d|", dfa[i*num_cols + j]);
            else
                printf(format_str, ".");
        }
        cout<<endl;
    }
}

void print_word_list(vector<string> word_list, int num, int total_words) {
    if(num > total_words){
        num = total_words;
    }
    cout<<"total number of words: "<<total_words<<endl;
    cout<<"printing words: "<<num<<endl;

    int start = 0;

    for(vector<string>::iterator it=word_list.begin(); start < num; ++it, start++) {
        cout<<*it<<endl;
    }
}

void print_state_outputs(int* dfa, string output[], int num_rows) {
    for(int i=0; i<num_rows; i++) {
        if(dfa[i*num_cols + 0]!=0) {
            cout<<"state:"<<i<<" "<<output[i]<<endl;
        }
    }
}

void print_fail_states(int fail_state[], int num_states) {
    for(int i=0; i<num_states; i++) {
        cout<<i<<" "<<fail_state[i]<<endl;
    }
}

void performance_test(int* dfa, int* fail_state, char* tweets, bool* valid_state, int num_tweets, int tweet_length, int num_of_words, int pattern_size) {
    GpuTimer timer;
    fflush(stdout);
    timer.Start();
    cout<<"SERIAL"<<","<<num_tweets<<","<<tweet_length<<","<<num_of_words<<","<<pattern_size<<",";
    profanity_filter_serial(dfa, fail_state, tweets, valid_state, num_tweets, tweet_length);
    timer.Stop();
    printf("%f\n", timer.Elapsed());
    memset(valid_state, false, num_tweets*sizeof(bool));
    fflush(stdout);

    timer.Start();
    cout<<"CUDA"<<","<<num_tweets<<","<<tweet_length<<","<<num_of_words<<","<<pattern_size<<",";
    profanity_filter_parallel(dfa, fail_state, tweets, valid_state, num_tweets, tweet_length, 1024, 256);
    timer.Stop();
    printf("%f\n", timer.Elapsed());

    fflush(stdout);
    timer.Start();
    cout<<"OpenACC"<<","<<num_tweets<<","<<tweet_length<<","<<num_of_words<<","<<pattern_size<<",";
    profanity_filter_acc_parallel(dfa, fail_state, tweets, valid_state, num_tweets, tweet_length);
    timer.Stop();
    printf("%f\n", timer.Elapsed());
}

void cuda_optimal_space_search(int* dfa, int* fail_state, char* tweets, bool* valid_state, int num_tweets, int tweet_length, int num_of_words, int num_threads, int num_blocks) {
    GpuTimer timer;
    fflush(stdout);
    timer.Start();
    profanity_filter_parallel(dfa, fail_state, tweets, valid_state, num_tweets, tweet_length, num_threads, num_blocks);
    timer.Stop();
    printf("%f", timer.Elapsed());
}

int main(int argc, char **argv) {

    bool lflag=false, tflag=false, sflag=false;
    int num_of_words;
    int c;
    int* fail_state;
    char* input_filename;
    int tweet_length;
    char* tweets;
    int* dfa;
    bool* valid_state;
    int num_tweets;
    int num_words_to_load;

    #ifdef _WIN32
    // Windows does not have getopt by default, so you may need to use a library or implement argument parsing differently
    // For now, we can simulate argument parsing or prompt user input
    // Here is a simple example to simulate -l option for Windows:
    if (argc > 1) {
        if (strcmp(argv[1], "-l") == 0 && argc > 2) {
            lflag = true;
            input_filename = argv[2];
        } else {
            print_usage();
            return 0;
        }
    } else {
        print_usage();
        return 0;
    }
    #else
    while((c = getopt(argc, argv, "l:ts?")) != -1) {
        switch(c) {
            case 'l':
                lflag = true;
                input_filename = optarg;
                break;

            case 't':
                tflag = true;
                break;

            case 's':
                sflag = true;
                break;

            case '?':
                print_usage();
                exit(0);
                break;

            default:
                exit(0);
                break;
        }
    }
    #endif

    if(!lflag) {
        cout<<"Please provide list of words to load"<<endl;
        exit(0);
    }

    if(sflag) {
        cout<<"# of records"<<","<<"# of characters in each record"<<","<<"# of patterns"<<","<<"Block size"<<","<<"# of threads"<<","<<"Runtime (in ms)"<<endl;
        for(int num_blocks=64; num_blocks<=256; num_blocks+=32) {
            for(int num_threads=128; num_threads<=1024; num_threads+= 64) {
                num_tweets = 1000000;
                tweet_length = 800;
                num_words_to_load = 800;
                vector<string> word_list;
                string output[num_rows];
                fail_state = (int *) malloc(num_rows*sizeof(int));
                dfa = (int *) malloc(num_rows*num_cols*sizeof(int));

                memset(dfa, 0, sizeof(int) * num_rows * num_cols);
                memset(fail_state, 0, sizeof(fail_state));

                word_list = load_input_file(input_filename, &num_of_words, num_words_to_load);

                generate_DFA(dfa, output, word_list);
                generate_fail_states(dfa, output, fail_state);

                valid_state = (bool *) malloc(num_tweets * sizeof(bool));

                tweets = load_tweets("data/tweets_big_800", num_tweets, tweet_length);

                memset(valid_state, false, sizeof(valid_state));
                cout<<num_tweets<<","<<tweet_length<<","<<num_words_to_load<<","<<num_blocks<<","<<num_threads<<",";
                cuda_optimal_space_search(dfa, fail_state, tweets, valid_state, num_tweets, tweet_length, num_words_to_load, num_threads, num_blocks);
                cout<<endl;
                free(dfa);
                free(fail_state);
                free(tweets);
                free(valid_state);
            }
        }

        exit(1);
    }
    cout<<"Type"<<","<<"# of records"<<","<<"# of characters in each record"<<","<<"# of patterns"<<","<<"Runtime (in ms)"<<endl;
    for( num_tweets=100000; num_tweets<=1000000; num_tweets+=100000) {
        for(tweet_length=100; tweet_length<=800; tweet_length+=100) {
            for(num_words_to_load=200; num_words_to_load<=800; num_words_to_load+=200) {
                vector<string> word_list;
                string output[num_rows];
                fail_state = (int *) malloc(num_rows*sizeof(int));
                dfa = (int *) malloc(num_rows*num_cols*sizeof(int));

                memset(dfa, 0, sizeof(int) * num_rows * num_cols);
                memset(fail_state, 0, sizeof(fail_state));

                word_list = load_input_file(input_filename, &num_of_words, num_words_to_load);

                generate_DFA(dfa, output, word_list);

                generate_fail_states(dfa, output, fail_state);

                valid_state = (bool *) malloc(num_tweets * sizeof(bool));

                if(tweet_length == 100)
                    tweets = load_tweets("data/tweets_big_100", num_tweets, tweet_length);
                if(tweet_length == 200)
                    tweets = load_tweets("data/tweets_big_200", num_tweets, tweet_length);
                if(tweet_length == 300)
                    tweets = load_tweets("data/tweets_big_300", num_tweets, tweet_length);
                if(tweet_length == 400)
                    tweets = load_tweets("data/tweets_big_400", num_tweets, tweet_length);
                if(tweet_length == 500)
                    tweets = load_tweets("data/tweets_big_500", num_tweets, tweet_length);
                if(tweet_length == 600)
                    tweets = load_tweets("data/tweets_big_600", num_tweets, tweet_length);
                if(tweet_length == 700)
                    tweets = load_tweets("data/tweets_big_700", num_tweets, tweet_length);
                if(tweet_length == 800)
                    tweets = load_tweets("data/tweets_big_800", num_tweets, tweet_length);

                memset(valid_state, false, sizeof(valid_state));

                performance_test(dfa, fail_state, tweets, valid_state, num_tweets, tweet_length, num_words_to_load, num_words_to_load);

                free(dfa);
                free(fail_state);
                free(tweets);
                free(valid_state);
            }
        }
    }

    return 1;
}
