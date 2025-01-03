"""
Generating data for SRNet
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License (see LICENSE for details)
Written by Yu Qian
"""

import os
import cv2
import cfg
from Synthtext.gen import datagen
from tqdm import tqdm

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    
    i_t_dir = os.path.join(cfg.data_dir, cfg.i_t_dir)
    i_s_dir = os.path.join(cfg.data_dir, cfg.i_s_dir)
    t_sk_dir = os.path.join(cfg.data_dir, cfg.t_sk_dir)
    s_sk_dir = os.path.join(cfg.data_dir, cfg.s_sk_dir)
    t_b_dir = os.path.join(cfg.data_dir, cfg.t_b_dir)

    
    makedirs(i_t_dir)
    makedirs(i_s_dir)
    makedirs(t_sk_dir)
    makedirs(s_sk_dir)
    makedirs(t_b_dir)

    i_s_txt_path = os.path.join(cfg.data_dir, 'i_s.txt')
    i_t_txt_path = os.path.join(cfg.data_dir, 'i_t.txt')

    gen = datagen()

    with open(i_s_txt_path, 'w') as i_s_txt_file, open(i_t_txt_path, 'w') as i_t_txt_file:
        for idx in tqdm(range(cfg.sample_num)):
            print("Generating step {:>6d} / {:>6d}".format(idx + 1, cfg.sample_num))
            while True:
                try:
                    i_s, s_sk, i_t, t_sk, t_b, s_text, t_text = gen.gen_srnet_data_with_background()
                    i_s_path = os.path.join(i_s_dir, str(idx) + '.png')
                    s_sk_path = os.path.join(s_sk_dir, str(idx) + '.png')
                    i_t_path = os.path.join(i_t_dir, str(idx) + '.png')
                    t_sk_path = os.path.join(t_sk_dir, str(idx) + '.png')
                    t_b_path = os.path.join(t_b_dir, str(idx) + '.png')
                    cv2.imwrite(i_s_path, i_s, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
                    cv2.imwrite(s_sk_path, s_sk, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
                    cv2.imwrite(i_t_path, i_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
                    cv2.imwrite(t_sk_path, t_sk, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
                    cv2.imwrite(t_b_path, t_b, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
                    i_s_txt_file.write(f"{str(idx)}.png {s_text}\n")
                    i_t_txt_file.write(f"{str(idx)}.png {t_text}\n")
                    break

                except Exception as e:
                    print(f"Error encountered at idx {idx}: {e}")
                    print("Retrying...")
                    continue 

if __name__ == '__main__':
    main()
