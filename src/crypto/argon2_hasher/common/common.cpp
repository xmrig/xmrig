//
// Created by Haifa Bogdan Adnan on 05/08/2018.
//

#include "DLLExport.h"
#include "common.h"
#include <dirent.h>

vector<string> getFiles(const string &folder) {
	vector<string> result;
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir (folder.c_str())) != NULL) {
		while ((ent = readdir (dir)) != NULL) {
			if(ent->d_type == DT_REG)
    			result.push_back(ent->d_name);
		}
		closedir (dir);
	}
	return result;
}
