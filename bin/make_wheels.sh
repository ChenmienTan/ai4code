! git clone https://mrcabbage972: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@github.com/mrcabbage972/ai4code.git
! pip freeze > reqs-existing.txt
! cd ai4code/src && python setup.py install
! pip freeze > reqs-package.txt
! grep -v -x -f reqs-existing.txt reqs-package.txt > reqs-final.txt
! cat reqs-final.txt
! cd ai4code/src && pip wheel . -r /kaggle/working/reqs-final.txt -w /kaggle/working/wheels --no-deps
! rm -r ai4code