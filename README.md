# Replication

A repo for replication.

Clone a repo into branch:

```bash
git clone git@github.com:User/RepoName.git

cd RepoName
git branch -m main RepoName
# git branch

git remote remove origin
git remote add origin git@github.com:X1AOX1A/Replication.git
# git remote -v

git push --set-upstream origin RepoName
```

To pull from original repo:

```bash
git config pull.rebase false 
git remote add RepoName git@github.com:User/RepoName.git
git pull RepoName main
```

To pull a specific branch:

```bash
mkdir RepoName
cd RepoName
git init
git branch -m master RepoName

git remote add origin git@github.com:X1AOX1A/Replication.git
git pull origin RepoName
git push --set-upstream origin RepoName
```
