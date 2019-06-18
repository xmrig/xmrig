# The MIT License (MIT)
#
# Copyright (c) .NET Foundation and Contributors
#
# All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# C to MASM include file translator
# This is replacement for the deprecated h2inc tool that used to be part of VS.

#
# The use of [console]::WriteLine (instead of Write-Output) is intentional.
# PowerShell 2.0 (installed by default on Windows 7) wraps lines written with
# Write-Output at whatever column width is being used by the current terminal,
# even when output is being redirected to a file. We can't have this behavior
# because it will cause the generated file to be malformed.
#

Function ProcessFile($filePath) {

    [console]::WriteLine("; File start: $filePath")

    Get-Content $filePath | ForEach-Object {
        
        if ($_ -match "^\s*#\spragma") {
            # Ignore pragmas
            return
        }
        
        if ($_ -match "^\s*#\s*include\s*`"(.*)`"")
        {
            # Expand includes.
            ProcessFile(Join-Path (Split-Path -Parent $filePath) $Matches[1])
            return
        }
        
        if ($_ -match "^\s*#define\s+(\S+)\s*(.*)")
        {
            # Augment #defines with their MASM equivalent
            $name = $Matches[1]
            $value = $Matches[2]

            # Note that we do not handle multiline constants
            
            # Strip comments from value
            $value = $value -replace "//.*", ""
            $value = $value -replace "/\*.*\*/", ""

            # Strip whitespaces from value
            $value = $value -replace "\s+$", ""

            # ignore #defines with arguments
            if ($name -notmatch "\(") {
                $HEX_NUMBER_PATTERN = "\b0x(\w+)\b"
                $DECIMAL_NUMBER_PATTERN = "(-?\b\d+\b)"
                       
                if ($value -match $HEX_NUMBER_PATTERN -or $value -match $DECIMAL_NUMBER_PATTERN) {
                    $value = $value -replace $HEX_NUMBER_PATTERN, "0`$1h"    # Convert hex constants
                    $value = $value -replace $DECIMAL_NUMBER_PATTERN, "`$1t" # Convert dec constants
                    [console]::WriteLine("$name EQU $value")
                } else {
                    [console]::WriteLine("$name TEXTEQU <$value>")
                }
            }            
        }
        
        # [console]::WriteLine("$_")
    }

    [console]::WriteLine("; File end: $filePath")
}

ProcessFile $args[0]
