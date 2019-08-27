#!/usr/bin/env node

'use strict';

const fs = require('fs');
const path = require('path');
const { text2h, addIncludes } = require('./js/opencl');
const cwd = process.cwd();


function cn()
{
    process.chdir(cwd);
    process.chdir(path.resolve('src/backend/opencl/cl/cn'));

    const cn = addIncludes('cryptonight.cl', [
        'algorithm.cl',
        'wolf-aes.cl',
        'wolf-skein.cl',
        'jh.cl',
        'blake256.cl',
        'groestl256.cl',
        'fast_int_math_v2.cl',
        'fast_div_heavy.cl'
    ]);

    //fs.writeFileSync('cryptonight_gen.cl', cn);
    fs.writeFileSync('cryptonight_cl.h', text2h(cn, 'xmrig', 'cryptonight_cl'));
}


cn();