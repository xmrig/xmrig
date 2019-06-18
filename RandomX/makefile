#CXX=g++-8
#CC=gcc-8
AR=gcc-ar
PLATFORM=$(shell uname -m)
CXXFLAGS=-std=c++11 -fPIC
CCFLAGS=-std=c99 -fPIC
ARFLAGS=rcs
BINDIR=bin
SRCDIR=src
TESTDIR=src/tests
OBJDIR=obj
LDFLAGS=-lpthread
RXA=$(BINDIR)/librandomx.a
BINARIES=$(RXA) $(BINDIR)/benchmark $(BINDIR)/code-generator
RXOBJS=$(addprefix $(OBJDIR)/,aes_hash.o argon2_ref.o dataset.o soft_aes.o virtual_memory.o vm_interpreted.o allocator.o assembly_generator_x86.o instruction.o randomx.o superscalar.o vm_compiled.o vm_interpreted_light.o argon2_core.o blake2_generator.o instructions_portable.o reciprocal.o virtual_machine.o vm_compiled_light.o blake2b.o)
ifeq ($(PLATFORM),amd64)
    RXOBJS += $(addprefix $(OBJDIR)/,jit_compiler_x86_static.o jit_compiler_x86.o)
    CXXFLAGS += -maes
endif
ifeq ($(PLATFORM),x86_64)
    RXOBJS += $(addprefix $(OBJDIR)/,jit_compiler_x86_static.o jit_compiler_x86.o)
    CXXFLAGS += -maes
endif

release: CXXFLAGS += -O3 -flto
release: CCFLAGS += -O3 -flto
release: LDFLAGS += -flto
release: $(BINARIES)

native: CXXFLAGS += -march=native -O3 -flto
native: CCFLAGS += -march=native -O3 -flto
native: $(BINARIES)

nolto: CXXFLAGS += -O3
nolto: CCFLAGS += -O3
nolto: $(BINARIES)

debug: CXXFLAGS += -g
debug: CCFLAGS += -g
debug: LDFLAGS += -g
debug: $(BINARIES)

profile: CXXFLAGS += -pg
profile: CCFLAGS += -pg
profile: LDFLAGS += -pg
profile: $(BINDIR)/benchmark

test: CXXFLAGS += -O0

$(RXA): $(RXOBJS) | $(BINDIR)
	$(AR) $(ARFLAGS) $@ $(RXOBJS)
$(OBJDIR):
	mkdir $(OBJDIR)
$(BINDIR):
	mkdir $(BINDIR)
$(OBJDIR)/benchmark.o: $(TESTDIR)/benchmark.cpp $(TESTDIR)/stopwatch.hpp \
 $(TESTDIR)/utility.hpp $(SRCDIR)/randomx.h $(SRCDIR)/blake2/endian.h
	$(CXX) $(CXXFLAGS) -pthread -c $< -o $@
$(BINDIR)/benchmark: $(OBJDIR)/benchmark.o $(RXA)
	$(CXX) $(LDFLAGS) -pthread $< $(RXA) -o $@
$(OBJDIR)/code-generator.o: $(TESTDIR)/code-generator.cpp $(TESTDIR)/utility.hpp \
 $(SRCDIR)/common.hpp $(SRCDIR)/blake2/endian.h \
 $(SRCDIR)/configuration.h $(SRCDIR)/randomx.h \
 $(SRCDIR)/assembly_generator_x86.hpp $(SRCDIR)/superscalar.hpp \
 $(SRCDIR)/superscalar_program.hpp $(SRCDIR)/instruction.hpp \
 $(SRCDIR)/blake2_generator.hpp $(SRCDIR)/aes_hash.hpp \
 $(SRCDIR)/blake2/blake2.h $(SRCDIR)/program.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
$(BINDIR)/code-generator: $(OBJDIR)/code-generator.o $(RXA)
	$(CXX) $(LDFLAGS) $< $(RXA) -o $@
$(OBJDIR)/aes_hash.o: $(SRCDIR)/aes_hash.cpp $(SRCDIR)/soft_aes.h $(SRCDIR)/intrin_portable.h | $(OBJDIR)
$(OBJDIR)/argon2_ref.o: $(SRCDIR)/argon2_ref.c $(SRCDIR)/argon2.h $(SRCDIR)/argon2_core.h \
 $(SRCDIR)/blake2/blamka-round-ref.h $(SRCDIR)/blake2/blake2.h \
 $(SRCDIR)/blake2/blake2-impl.h $(SRCDIR)/blake2/endian.h $(SRCDIR)/blake2/blake2-impl.h \
 $(SRCDIR)/blake2/blake2.h
$(OBJDIR)/blake2b.o: $(SRCDIR)/blake2/blake2b.c $(SRCDIR)/blake2/blake2.h \
 $(SRCDIR)/blake2/blake2-impl.h $(SRCDIR)/blake2/endian.h
	$(CC) $(CCFLAGS) -c $< -o $@
$(OBJDIR)/dataset.o: $(SRCDIR)/dataset.cpp $(SRCDIR)/common.hpp $(SRCDIR)/blake2/endian.h \
 $(SRCDIR)/configuration.h $(SRCDIR)/randomx.h $(SRCDIR)/dataset.hpp \
 $(SRCDIR)/superscalar_program.hpp $(SRCDIR)/instruction.hpp $(SRCDIR)/jit_compiler_x86.hpp \
 $(SRCDIR)/allocator.hpp $(SRCDIR)/virtual_memory.hpp $(SRCDIR)/superscalar.hpp \
 $(SRCDIR)/blake2_generator.hpp $(SRCDIR)/reciprocal.h $(SRCDIR)/argon2.h $(SRCDIR)/argon2_core.h \
 $(SRCDIR)/intrin_portable.h
$(OBJDIR)/jit_compiler_x86.o: $(SRCDIR)/jit_compiler_x86.cpp $(SRCDIR)/jit_compiler_x86.hpp \
 $(SRCDIR)/common.hpp $(SRCDIR)/blake2/endian.h $(SRCDIR)/configuration.h $(SRCDIR)/randomx.h \
 $(SRCDIR)/jit_compiler_x86_static.hpp $(SRCDIR)/superscalar.hpp \
 $(SRCDIR)/superscalar_program.hpp $(SRCDIR)/instruction.hpp $(SRCDIR)/blake2_generator.hpp \
 $(SRCDIR)/program.hpp $(SRCDIR)/reciprocal.h $(SRCDIR)/virtual_memory.hpp \
 $(SRCDIR)/instruction_weights.hpp
$(OBJDIR)/jit_compiler_x86_static.o: $(SRCDIR)/jit_compiler_x86_static.S $(SRCDIR)/configuration.h \
 $(SRCDIR)/asm/program_prologue_linux.inc $(SRCDIR)/asm/program_xmm_constants.inc \
 $(SRCDIR)/asm/program_loop_load.inc $(SRCDIR)/asm/program_read_dataset.inc \
 $(SRCDIR)/asm/program_read_dataset_sshash_init.inc \
 $(SRCDIR)/asm/program_read_dataset_sshash_fin.inc \
 $(SRCDIR)/asm/program_loop_store.inc $(SRCDIR)/asm/program_epilogue_linux.inc \
 $(SRCDIR)/asm/program_epilogue_store.inc $(SRCDIR)/asm/program_sshash_load.inc \
 $(SRCDIR)/asm/program_sshash_prefetch.inc $(SRCDIR)/asm/program_sshash_constants.inc \
 $(SRCDIR)/asm/randomx_reciprocal.inc
$(OBJDIR)/soft_aes.o: $(SRCDIR)/soft_aes.cpp $(SRCDIR)/soft_aes.h $(SRCDIR)/intrin_portable.h
$(OBJDIR)/virtual_memory.o: $(SRCDIR)/virtual_memory.cpp $(SRCDIR)/virtual_memory.hpp
$(OBJDIR)/vm_interpreted.o: $(SRCDIR)/vm_interpreted.cpp $(SRCDIR)/vm_interpreted.hpp \
 $(SRCDIR)/common.hpp $(SRCDIR)/blake2/endian.h $(SRCDIR)/configuration.h $(SRCDIR)/randomx.h \
 $(SRCDIR)/virtual_machine.hpp $(SRCDIR)/program.hpp $(SRCDIR)/instruction.hpp \
 $(SRCDIR)/intrin_portable.h $(SRCDIR)/allocator.hpp $(SRCDIR)/dataset.hpp \
 $(SRCDIR)/superscalar_program.hpp $(SRCDIR)/jit_compiler_x86.hpp $(SRCDIR)/reciprocal.h \
 $(SRCDIR)/instruction_weights.hpp
$(OBJDIR)/allocator.o: $(SRCDIR)/allocator.cpp $(SRCDIR)/allocator.hpp $(SRCDIR)/intrin_portable.h \
 $(SRCDIR)/virtual_memory.hpp $(SRCDIR)/common.hpp $(SRCDIR)/blake2/endian.h \
 $(SRCDIR)/configuration.h $(SRCDIR)/randomx.h
$(OBJDIR)/assembly_generator_x86.o: $(SRCDIR)/assembly_generator_x86.cpp \
 $(SRCDIR)/assembly_generator_x86.hpp $(SRCDIR)/common.hpp $(SRCDIR)/blake2/endian.h \
 $(SRCDIR)/configuration.h $(SRCDIR)/randomx.h $(SRCDIR)/reciprocal.h $(SRCDIR)/program.hpp \
 $(SRCDIR)/instruction.hpp $(SRCDIR)/superscalar.hpp $(SRCDIR)/superscalar_program.hpp \
 $(SRCDIR)/blake2_generator.hpp $(SRCDIR)/instruction_weights.hpp
$(OBJDIR)/instruction.o: $(SRCDIR)/instruction.cpp $(SRCDIR)/instruction.hpp \
 $(SRCDIR)/blake2/endian.h $(SRCDIR)/common.hpp $(SRCDIR)/configuration.h $(SRCDIR)/randomx.h \
 $(SRCDIR)/instruction_weights.hpp
$(OBJDIR)/randomx.o: $(SRCDIR)/randomx.cpp $(SRCDIR)/randomx.h $(SRCDIR)/dataset.hpp $(SRCDIR)/common.hpp \
 $(SRCDIR)/blake2/endian.h $(SRCDIR)/configuration.h $(SRCDIR)/superscalar_program.hpp \
 $(SRCDIR)/instruction.hpp $(SRCDIR)/jit_compiler_x86.hpp $(SRCDIR)/allocator.hpp \
 $(SRCDIR)/vm_interpreted.hpp $(SRCDIR)/virtual_machine.hpp $(SRCDIR)/program.hpp \
 $(SRCDIR)/intrin_portable.h $(SRCDIR)/vm_interpreted_light.hpp $(SRCDIR)/vm_compiled.hpp \
 $(SRCDIR)/vm_compiled_light.hpp $(SRCDIR)/blake2/blake2.h
$(OBJDIR)/superscalar.o: $(SRCDIR)/superscalar.cpp $(SRCDIR)/configuration.h $(SRCDIR)/program.hpp \
 $(SRCDIR)/common.hpp $(SRCDIR)/blake2/endian.h $(SRCDIR)/randomx.h $(SRCDIR)/instruction.hpp \
 $(SRCDIR)/superscalar.hpp $(SRCDIR)/superscalar_program.hpp $(SRCDIR)/blake2_generator.hpp \
 $(SRCDIR)/intrin_portable.h $(SRCDIR)/reciprocal.h
$(OBJDIR)/vm_compiled.o: $(SRCDIR)/vm_compiled.cpp $(SRCDIR)/vm_compiled.hpp \
 $(SRCDIR)/virtual_machine.hpp $(SRCDIR)/common.hpp $(SRCDIR)/blake2/endian.h \
 $(SRCDIR)/configuration.h $(SRCDIR)/randomx.h $(SRCDIR)/program.hpp $(SRCDIR)/instruction.hpp \
 $(SRCDIR)/jit_compiler_x86.hpp $(SRCDIR)/allocator.hpp $(SRCDIR)/dataset.hpp \
 $(SRCDIR)/superscalar_program.hpp
$(OBJDIR)/vm_interpreted_light.o: $(SRCDIR)/vm_interpreted_light.cpp \
 $(SRCDIR)/vm_interpreted_light.hpp $(SRCDIR)/vm_interpreted.hpp $(SRCDIR)/common.hpp \
 $(SRCDIR)/blake2/endian.h $(SRCDIR)/configuration.h $(SRCDIR)/randomx.h \
 $(SRCDIR)/virtual_machine.hpp $(SRCDIR)/program.hpp $(SRCDIR)/instruction.hpp \
 $(SRCDIR)/intrin_portable.h $(SRCDIR)/allocator.hpp $(SRCDIR)/dataset.hpp \
 $(SRCDIR)/superscalar_program.hpp $(SRCDIR)/jit_compiler_x86.hpp
$(OBJDIR)/argon2_core.o: $(SRCDIR)/argon2_core.c $(SRCDIR)/argon2_core.h $(SRCDIR)/argon2.h \
 $(SRCDIR)/blake2/blake2.h $(SRCDIR)/blake2/blake2-impl.h $(SRCDIR)/blake2/endian.h
$(OBJDIR)/blake2_generator.o: $(SRCDIR)/blake2_generator.cpp $(SRCDIR)/blake2/blake2.h \
 $(SRCDIR)/blake2/endian.h $(SRCDIR)/blake2_generator.hpp
$(OBJDIR)/instructions_portable.o: $(SRCDIR)/instructions_portable.cpp $(SRCDIR)/common.hpp \
 $(SRCDIR)/blake2/endian.h $(SRCDIR)/configuration.h $(SRCDIR)/randomx.h \
 $(SRCDIR)/intrin_portable.h
$(OBJDIR)/reciprocal.o: $(SRCDIR)/reciprocal.c $(SRCDIR)/reciprocal.h
$(OBJDIR)/virtual_machine.o: $(SRCDIR)/virtual_machine.cpp $(SRCDIR)/virtual_machine.hpp \
 $(SRCDIR)/common.hpp $(SRCDIR)/blake2/endian.h $(SRCDIR)/configuration.h $(SRCDIR)/randomx.h \
 $(SRCDIR)/program.hpp $(SRCDIR)/instruction.hpp $(SRCDIR)/aes_hash.hpp $(SRCDIR)/blake2/blake2.h \
 $(SRCDIR)/intrin_portable.h $(SRCDIR)/allocator.hpp
$(OBJDIR)/vm_compiled_light.o: $(SRCDIR)/vm_compiled_light.cpp $(SRCDIR)/vm_compiled_light.hpp \
 $(SRCDIR)/vm_compiled.hpp $(SRCDIR)/virtual_machine.hpp $(SRCDIR)/common.hpp \
 $(SRCDIR)/blake2/endian.h $(SRCDIR)/configuration.h $(SRCDIR)/randomx.h $(SRCDIR)/program.hpp \
 $(SRCDIR)/instruction.hpp $(SRCDIR)/jit_compiler_x86.hpp $(SRCDIR)/allocator.hpp \
 $(SRCDIR)/dataset.hpp $(SRCDIR)/superscalar_program.hpp

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CCFLAGS) -c $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.S
	$(CXX) -x assembler-with-cpp -c $< -o $@

clean:
	rm -f $(BINARIES) $(OBJDIR)/*.o
