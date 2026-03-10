#!/usr/bin/env node

'use strict';

const fs = require('fs');
const path = require('path');
const { text2h, text2h_bundle, addIncludes } = require('./js/opencl');
const { opencl_minify } = require('./js/opencl_minify');


function cn()
{
    const cn = opencl_minify(addIncludes('cryptonight.cl', [
        'algorithm.cl',
        'wolf-aes.cl',
        'wolf-skein.cl',
        'jh.cl',
        'blake256.cl',
        'groestl256.cl',
        'fast_int_math_v2.cl',
        'fast_div_heavy.cl',
        'keccak.cl'
    ]));

    // fs.writeFileSync('cryptonight_gen.cl', cn);
    fs.writeFileSync('cryptonight_cl.h', text2h(cn, 'xmrig', 'cryptonight_cl'));
}


function cn_r()
{
    const items = {};

    items.cryptonight_r_defines_cl = opencl_minify(addIncludes('cryptonight_r_defines.cl', [ 'wolf-aes.cl' ]));
    items.cryptonight_r_cl         = opencl_minify(fs.readFileSync('cryptonight_r.cl', 'utf8'));

    // for (let key in items) {
    //      fs.writeFileSync(key + '_gen.cl', items[key]);
    // }

    fs.writeFileSync('cryptonight_r_cl.h', text2h_bundle('xmrig', items));
}


function rx()
{
    let rx = addIncludes('randomx.cl', [
        '../cn/algorithm.cl',
        'randomx_constants_monero.h',
        'randomx_constants_wow.h',
        'randomx_constants_arqma.h',
        'randomx_constants_graft.h',
        'aes.cl',
        'blake2b.cl',
        'randomx_vm.cl',
        'randomx_jit.cl'
    ]);

    rx = rx.replace(/(\t| )*#include "fillAes1Rx4.cl"/g, fs.readFileSync('fillAes1Rx4.cl', 'utf8'));
    rx = rx.replace(/(\t| )*#include "blake2b_double_block.cl"/g, fs.readFileSync('blake2b_double_block.cl', 'utf8'));
    rx = opencl_minify(rx);

    //fs.writeFileSync('randomx_gen.cl', rx);
    fs.writeFileSync('randomx_cl.h', text2h(rx, 'xmrig', 'randomx_cl'));
}


function kawpow()
{
    const kawpow = opencl_minify(addIncludes('kawpow.cl', [ 'defs.h' ]));
    const kawpow_dag = opencl_minify(addIncludes('kawpow_dag.cl', [ 'defs.h' ]));

    // fs.writeFileSync('kawpow_gen.cl', kawpow);
    fs.writeFileSync('kawpow_cl.h', text2h(kawpow, 'xmrig', 'kawpow_cl'));
    fs.writeFileSync('kawpow_dag_cl.h', text2h(kawpow_dag, 'xmrig', 'kawpow_dag_cl'));
}

for (let i = 0; i < 2; i++) {
    if (fs.existsSync('src/backend/opencl/cl/OclSource.h')) {
        break;
    }

    process.chdir('..');
}

process.chdir(path.resolve('src/backend/opencl/cl'));

const cwd = process.cwd();

process.chdir(path.resolve(cwd, 'cn'));
cn();
cn_r();

process.chdir(path.resolve(cwd, 'rx'));
rx();

process.chdir(path.resolve(cwd, 'kawpow'));
kawpow();
