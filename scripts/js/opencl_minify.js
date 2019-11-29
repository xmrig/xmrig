'use strict';

function opencl_minify(input)
{
    let out = input.replace(/\r/g, '');
    out = out.replace(/\/\*[\s\S]*?\*\/|\/\/.*$/gm, ''); // comments
    out = out.replace(/^#\s+/gm, '#');        // macros with spaces
    out = out.replace(/\n{2,}/g, '\n');       // empty lines
    out = out.replace(/^\s+/gm, '');          // leading whitespace
    out = out.replace(/ {2,}/g, ' ');         // extra whitespace

    let array = out.split('\n').map(line => {
        if (line[0] === '#') {
            return line;
        }

        line = line.replace(/, /g, ',');
        line = line.replace(/ \? /g, '?');
        line = line.replace(/ : /g, ':');
        line = line.replace(/ = /g, '=');
        line = line.replace(/ != /g, '!=');
        line = line.replace(/ >= /g, '>=');
        line = line.replace(/ <= /g, '<=');
        line = line.replace(/ == /g, '==');
        line = line.replace(/ \+= /g, '+=');
        line = line.replace(/ -= /g, '-=');
        line = line.replace(/ \|= /g, '|=');
        line = line.replace(/ \| /g, '|');
        line = line.replace(/ \|\| /g, '||');
        line = line.replace(/ & /g, '&');
        line = line.replace(/ && /g, '&&');
        line = line.replace(/ > /g, '>');
        line = line.replace(/ < /g, '<');
        line = line.replace(/ \+ /g, '+');
        line = line.replace(/ - /g, '-');
        line = line.replace(/ \* /g, '*');
        line = line.replace(/ \^ /g, '^');
        line = line.replace(/ & /g, '&');
        line = line.replace(/ \/ /g, '/');
        line = line.replace(/ << /g, '<<');
        line = line.replace(/ >> /g, '>>');
        line = line.replace(/if \(/g, 'if(');

        return line;
    });

    return array.join('\n');
}


module.exports.opencl_minify = opencl_minify;