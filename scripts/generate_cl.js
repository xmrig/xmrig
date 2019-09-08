#!/usr/bin/env node

'use strict';

const fs = require('fs');
const path = require('path');
const { text2h, text2h_bundle, addIncludes } = require('./js/opencl');
// const cwd = process.cwd();


function cn()
{
    const cn = addIncludes('cryptonight.cl', [
        'algorithm.cl',
        'wolf-aes.cl',
        'wolf-skein.cl',
        'jh.cl',
        'blake256.cl',
        'groestl256.cl',
        'fast_int_math_v2.cl',
        'fast_div_heavy.cl',
        'keccak.cl'
    ]);

    //fs.writeFileSync('cryptonight_gen.cl', cn);
    fs.writeFileSync('cryptonight_cl.h', text2h(cn, 'xmrig', 'cryptonight_cl'));
}


function cn_r()
{
    const items = {};

    items.cryptonight_r_defines_cl = addIncludes('cryptonight_r_defines.cl', [ 'wolf-aes.cl' ]);
    items.cryptonight_r_cl         = fs.readFileSync('cryptonight_r.cl', 'utf8');

    // for (let key in items) {
    //     fs.writeFileSync(key + '_gen.cl', items[key]);
    // }

    fs.writeFileSync('cryptonight_r_cl.h', text2h_bundle('xmrig', items));
}


function cn_gpu()
{
    const cn_gpu = addIncludes('cryptonight_gpu.cl', [ 'wolf-aes.cl', 'keccak.cl' ]);

    //fs.writeFileSync('cryptonight_gpu_gen.cl', cn_gpu);
    fs.writeFileSync('cryptonight_gpu_cl.h', text2h(cn_gpu, 'xmrig', 'cryptonight_gpu_cl'));
}


process.chdir(path.resolve('src/backend/opencl/cl/cn'));

cn();
cn_r();
cn_gpu();