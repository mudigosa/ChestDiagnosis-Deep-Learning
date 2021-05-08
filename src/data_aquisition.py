import sys
import os
import urllib.request
import tarfile

###
def download_tar_files():
    # URLs for the zip files
    links = [
        'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
        'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
        'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
        'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
        'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
        'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
        'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
        'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
        'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
        'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
        'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
        'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
    ]

    for idx, link in enumerate(links):
        fn = 'images_%02d.tar.gz' % (idx+1)
        print('downloading '+fn+'...')
        urllib.request.urlretrieve(link, fn)  # download the zip file

    print("Download complete. Please check the checksums")
    return None
####
def extarct_images_from_tar_files(n_files=12):
    for i in range(n_files):
        filenumber = i+1
        fn = 'images_%02d.tar.gz' % (filenumber)
        print('extracting '+fn+'...')
        tar = tarfile.open(fn, "r:gz")
        tar.extractall()
        tar.close()
    print('Extraction Complete')

if __name__ == '__main__':
    
    if len(sys.argv) == 2:
        directory = sys.argv[1]
        
        # Set the directory to download data and extract images
        os.chdir(directory)

        # Download the tar files from NIH website
        download_tar_files()

        # Extract images from the tar files
        extarct_images_from_tar_files()
    else:
        print('Provide the directory to download and extract images')