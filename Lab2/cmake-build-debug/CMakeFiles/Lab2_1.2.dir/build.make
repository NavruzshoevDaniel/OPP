# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = "/Users/daniel/Library/Application Support/JetBrains/Toolbox/apps/CLion/ch-0/193.6494.38/CLion.app/Contents/bin/cmake/mac/bin/cmake"

# The command to remove a file.
RM = "/Users/daniel/Library/Application Support/JetBrains/Toolbox/apps/CLion/ch-0/193.6494.38/CLion.app/Contents/bin/cmake/mac/bin/cmake" -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/daniel/CLionProjects/OPP/Lab2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/daniel/CLionProjects/OPP/Lab2/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/Lab2_1.2.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Lab2_1.2.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Lab2_1.2.dir/flags.make

CMakeFiles/Lab2_1.2.dir/mainOpenMP2.cpp.o: CMakeFiles/Lab2_1.2.dir/flags.make
CMakeFiles/Lab2_1.2.dir/mainOpenMP2.cpp.o: ../mainOpenMP2.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/daniel/CLionProjects/OPP/Lab2/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Lab2_1.2.dir/mainOpenMP2.cpp.o"
	/usr/local/bin/g++-9  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Lab2_1.2.dir/mainOpenMP2.cpp.o -c /Users/daniel/CLionProjects/OPP/Lab2/mainOpenMP2.cpp

CMakeFiles/Lab2_1.2.dir/mainOpenMP2.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Lab2_1.2.dir/mainOpenMP2.cpp.i"
	/usr/local/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/daniel/CLionProjects/OPP/Lab2/mainOpenMP2.cpp > CMakeFiles/Lab2_1.2.dir/mainOpenMP2.cpp.i

CMakeFiles/Lab2_1.2.dir/mainOpenMP2.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Lab2_1.2.dir/mainOpenMP2.cpp.s"
	/usr/local/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/daniel/CLionProjects/OPP/Lab2/mainOpenMP2.cpp -o CMakeFiles/Lab2_1.2.dir/mainOpenMP2.cpp.s

# Object files for target Lab2_1.2
Lab2_1_2_OBJECTS = \
"CMakeFiles/Lab2_1.2.dir/mainOpenMP2.cpp.o"

# External object files for target Lab2_1.2
Lab2_1_2_EXTERNAL_OBJECTS =

Lab2_1.2: CMakeFiles/Lab2_1.2.dir/mainOpenMP2.cpp.o
Lab2_1.2: CMakeFiles/Lab2_1.2.dir/build.make
Lab2_1.2: CMakeFiles/Lab2_1.2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/daniel/CLionProjects/OPP/Lab2/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Lab2_1.2"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Lab2_1.2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Lab2_1.2.dir/build: Lab2_1.2

.PHONY : CMakeFiles/Lab2_1.2.dir/build

CMakeFiles/Lab2_1.2.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Lab2_1.2.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Lab2_1.2.dir/clean

CMakeFiles/Lab2_1.2.dir/depend:
	cd /Users/daniel/CLionProjects/OPP/Lab2/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/daniel/CLionProjects/OPP/Lab2 /Users/daniel/CLionProjects/OPP/Lab2 /Users/daniel/CLionProjects/OPP/Lab2/cmake-build-debug /Users/daniel/CLionProjects/OPP/Lab2/cmake-build-debug /Users/daniel/CLionProjects/OPP/Lab2/cmake-build-debug/CMakeFiles/Lab2_1.2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Lab2_1.2.dir/depend

