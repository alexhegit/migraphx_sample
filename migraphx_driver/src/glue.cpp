/*
 * glue.cpp - read and return strings for GLUE benchmark data
 *
 */
#include <iostream>
#include <fstream>
#include <string>
#include "migx.hpp"
#include <sys/stat.h>

struct glue_config {
  enum glue_type gtype;
  std::string val; // filename, relative to glue_dir
  int label_field; // TSV column
  int sent1_field; // TSV column
  int sent2_field; // TSV column
} glue_config_db[] = {
  { glue_cola, std::string("/CoLA/dev.tsv"), 2, 4, -1 },
};
int num_glue_config = sizeof(glue_config_db)/sizeof(glue_config_db[0]);

int dump_glue(enum glue_type gtype, std::string glue_dir){
  struct stat statbuf;

  if (glue_dir.empty() || stat(glue_dir.c_str(),&statbuf)){
    std::cerr << migx_program << ": invalid --gluedir" << std::endl;
    return 1;
  } else if (!S_ISDIR(statbuf.st_mode)){
    std::cerr << migx_program << ": invalid --gluedir, not a directory" << std::endl;
    return 1;
  }

  int glue_idx;
  bool found = false;
  for (glue_idx = 0;glue_idx < num_glue_config; glue_idx++){
    if (glue_config_db[glue_idx].gtype == gtype){
      found = true;
      break;
    }
  }

  if (!found){
    std::cerr << migx_program << ": --glue= test type not implemented" << std::endl;
    return 1;
  }
  return 0;
}
