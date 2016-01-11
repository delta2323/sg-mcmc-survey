import datetime
import glob
import os
import shutil

import maf
import maflib


top = '.'
out = 'build.{}'.format(
    datetime.datetime.now().strftime('%y%m%d-%H%M%S'))
ROOT = os.getcwd()


@maflib.util.rule
def copy_scripts(task):
    scripts = glob.glob('{}/*.py'.format(ROOT))
    scripts.append('{}/wscript'.format(ROOT))
    out_dir = task.outputs[0].abspath()
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for s in scripts:
        shutil.copy(s, out_dir)


def configure(conf):
    pass


def build(exp):
    exp(target='scripts/',
        rule=copy_scripts)

    exp(target='hmc/visualize.png hmc/log.txt',
        rule='python %s/toy_hmc.py '
        '--visualize ${TGT[0].abspath()} '
        '> ${TGT[1].abspath()}' % (ROOT))

    exp(target='sgld/visualize.png sgld/log.txt',
        rule='python %s/toy_sgld.py '
        '--visualize ${TGT[0].abspath()} '
        '> ${TGT[1].abspath()}' % (ROOT))

    exp(target='sghmc/visualize.png sghmc/log.txt',
        parameters=maflib.util.product(
            {'F': [1.0, 10.0], 'D': [1.0]}
            ),
        rule='python %s/toy_sghmc.py '
        '--visualize ${TGT[0].abspath()} '
        '--F ${F} --D ${D}'
        '> ${TGT[1].abspath()}' % (ROOT))

    exp(target='msgnht/visualize.png msgnht/log.txt',
        parameters=maflib.util.product({'D': [1.0, 10]}),
        rule='python %s/toy_sghmc.py '
        '--visualize ${TGT[0].abspath()} --D ${D} '
        '> ${TGT[1].abspath()}' % (ROOT))
