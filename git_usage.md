# how to fork gitlab repository

https://gist.github.com/DavideMontersino/810ebaa170a2aa2d2cad

# how to sync a fork

https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork

Steps:
- fetch from remote forked repository (upstream/master)
- merge it to the local repository (master)
- push it to the remote forking repository (origin/master)

```
git fetch upstream
git checkout master
git merge upstream/master
git push origin
```