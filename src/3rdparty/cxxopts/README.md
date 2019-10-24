[![Build Status](https://travis-ci.org/jarro2783/cxxopts.svg?branch=master)](https://travis-ci.org/jarro2783/cxxopts)

# Release versions

Note that `master` is generally a work in progress, and you probably want to use a
tagged release version.

# Quick start

This is a lightweight C++ option parser library, supporting the standard GNU
style syntax for options.

Options can be given as:

    --long
    --long=argument
    --long argument
    -a
    -ab
    -abc argument

where c takes an argument, but a and b do not.

Additionally, anything after `--` will be parsed as a positional argument.

## Basics

    #include <cxxopts.hpp>

Create a cxxopts::Options instance.

    cxxopts::Options options("MyProgram", "One line description of MyProgram");

Then use `add_options`.

    options.add_options()
      ("d,debug", "Enable debugging")
      ("f,file", "File name", cxxopts::value<std::string>())
      ;

Options are declared with a long and an optional short option. A description
must be provided. The third argument is the value, if omitted it is boolean.
Any type can be given as long as it can be parsed, with operator>>.

To parse the command line do:

    auto result = options.parse(argc, argv);

To retrieve an option use `result.count("option")` to get the number of times
it appeared, and

    result["opt"].as<type>()

to get its value. If "opt" doesn't exist, or isn't of the right type, then an
exception will be thrown.

Note that the result of `options.parse` should only be used as long as the
`options` object that created it is in scope.

## Exceptions

Exceptional situations throw C++ exceptions. There are two types of
exceptions: errors defining the options, and errors when parsing a list of
arguments. All exceptions derive from `cxxopts::OptionException`. Errors
defining options derive from `cxxopts::OptionSpecException` and errors
parsing arguments derive from `cxxopts::OptionParseException`.

All exceptions define a `what()` function to get a printable string
explaining the error.

## Help groups

Options can be placed into groups for the purposes of displaying help messages.
To place options in a group, pass the group as a string to `add_options`. Then,
when displaying the help, pass the groups that you would like displayed as a
vector to the `help` function.

## Positional Arguments

Positional arguments can be optionally parsed into one or more options.
To set up positional arguments, call

    options.parse_positional({"first", "second", "last"})

where "last" should be the name of an option with a container type, and the
others should have a single value.

## Default and implicit values

An option can be declared with a default or an implicit value, or both.

A default value is the value that an option takes when it is not specified
on the command line. The following specifies a default value for an option:

    cxxopts::value<std::string>()->default_value("value")

An implicit value is the value that an option takes when it is given on the
command line without an argument. The following specifies an implicit value:

    cxxopts::value<std::string>()->implicit_value("implicit")

If an option had both, then not specifying it would give the value `"value"`,
writing it on the command line as `--option` would give the value `"implicit"`,
and writing `--option=another` would give it the value `"another"`.

Note that the default and implicit value is always stored as a string,
regardless of the type that you want to store it in. It will be parsed as
though it was given on the command line.

## Boolean values

Boolean options have a default implicit value of `"true"`, which can be
overridden. The effect is that writing `-o` by itself will set option `o` to
`true`. However, they can also be written with various strings using `=value`.
There is no way to disambiguate positional arguments from the value following
a boolean, so we have chosen that they will be positional arguments, and
therefore, `-o false` does not work.

## `std::vector<T>` values

Parsing of list of values in form of an `std::vector<T>` is also supported, as long as `T`
can be parsed. To separate single values in a list the definition `CXXOPTS_VECTOR_DELIMITER`
is used, which is ',' by default. Ensure that you use no whitespaces between values because
those would be interpreted as the next command line option. Example for a command line option
that can be parsed as a `std::vector<double>`:

~~~
--my_list=1,-2.1,3,4.5
~~~

## Custom help

The string after the program name on the first line of the help can be
completely replaced by calling `options.custom_help`. Note that you might
also want to override the positional help by calling `options.positional_help`.

# Linking

This is a header only library.

# Requirements

The only build requirement is a C++ compiler that supports C++11 features such as:

* regex
* constexpr
* default constructors

GCC >= 4.9 or clang >= 3.1 with libc++ are known to work.

The following compilers are known not to work:

* MSVC 2013

# TODO list

* Allow unrecognised options.
