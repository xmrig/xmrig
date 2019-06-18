/*
Copyright (c) 2018-2019, tevador <tevador@gmail.com>

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
	* Redistributions of source code must retain the above copyright
	  notice, this list of conditions and the following disclaimer.
	* Redistributions in binary form must reproduce the above copyright
	  notice, this list of conditions and the following disclaimer in the
	  documentation and/or other materials provided with the distribution.
	* Neither the name of the copyright holder nor the
	  names of its contributors may be used to endorse or promote products
	  derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <chrono>
#include <cstdint>

class Stopwatch {
public:
	Stopwatch(bool startNow = false) {
		reset();
		if (startNow) {
			start();
		}
	}
	void reset() {
		isRunning = false;
		elapsed = 0;
	}
	void start() {
		if (!isRunning) {
			startMark = std::chrono::high_resolution_clock::now();
			isRunning = true;
		}
	}
	void restart() {
		startMark = std::chrono::high_resolution_clock::now();
		isRunning = true;
		elapsed = 0;
	}
	void stop() {
		if (isRunning) {
			chrono_t endMark = std::chrono::high_resolution_clock::now();
			uint64_t ns = std::chrono::duration_cast<sw_unit>(endMark - startMark).count();
			elapsed += ns;
			isRunning = false;
		}
	}
	double getElapsed() const {
		return getElapsedNanosec() / 1e+9;
	}
private:
	using chrono_t = std::chrono::high_resolution_clock::time_point;
	using sw_unit = std::chrono::nanoseconds;
	chrono_t startMark;
	uint64_t elapsed;
	bool isRunning;

	uint64_t getElapsedNanosec() const {
		uint64_t elns = elapsed;
		if (isRunning) {
			chrono_t endMark = std::chrono::high_resolution_clock::now();
			uint64_t ns = std::chrono::duration_cast<sw_unit>(endMark - startMark).count();
			elns += ns;
		}
		return elns;
	}
};