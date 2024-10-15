# Organizing Software Development

There are a lot's of buzzwords like test-driver development, or continuos integration/deployment/profiling, or agile, but as always, everyone understands them differently.
Here is our structure for organizing software development.

1. If you want to add/edit/remove code, first, open an issue.
2. Every issue, has gets it's own branch in the same repo, generally forked from `main-dev`.
  1. If you want your every commit displayed and discussed, push right there.
  2. If you want to work on something privately, fork it under your own username.
3. For your Pull Request to pass a review, submit not just the code, but also the tests, and the documentation.

Sounds simple.
Now let's dive into details.

## Git Credentials

- `git config --global user.email "id+username@users.noreply.github.com"`

Git is amazing.
It allows navigating the history, ~~to punish the guilty, and to liberate the oppressed~~...
It doesn't work, however, if your `git log` looks like this:

```txt
commit e5ca9b4b78bfe359a1a293c5cba0526eed02ec54
Author: Ash <ash@localhost>
Date:   Tue Jun 14 18:05:38 2022 +0400
```

The obvious recommendation is to use real full names and reachable emails, instead of `localhost` names.
But there is probably a better way.
The GitHub profile is the face of the software developer.
If you want to build a career, select a nickname, register a GitHub account, and you will get a unique ID.
Concatenating those, you will get a special GitHub address, that I use for all my commits.

```txt
commit e5ca9b4b78bfe359a1a293c5cba0526eed02ec54
Author: Ashot Vardanian <1983160+ashvardanian@users.noreply.github.com>
Date:   Tue Jun 14 18:05:38 2022 +0400
```

> Configuring Git credentials.

## Branches

- `main` for last stable version.
- `main-dev` to synchronize the development.
- `150-feature-add-something`-like branch per issue.
- `v1.0.0`-like branch per LTS version.

Since ~~the dawn of time~~ around 2020, the default branch on Github is called `main`, and not `master`.
That is where the most recent stable version of your software lives.

The `trunk`, `development`, or shorter `dev`, are generally the most active parts of your project.
In our case, we call that branch `main-dev`, because of the way branch protection rules are defined for GitHub.
All the `main*` branches we protect.
Currently with the following rules:

- [x] Require Pull Request Before Merging
  - [x] Require approvals
  - [ ] Dismiss stale pull request approvals when new commits are pushed
  - [x] Require review from Code Owners
  - [ ] Restrict who can dismiss pull request reviews
  - [x] Allow specified actors to bypass required pull requests: core maintainers
  - [ ] Require approval of the most recent reviewable push
- [ ] Require status checks to pass before merging [info](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches#require-status-checks-before-merging)
- [x] Require conversation resolution before merging [info](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches#require-conversation-resolution-before-merging)
- [ ] Require signed commits [info](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches#require-signed-commits)
- [ ] Require linear history [info](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches#require-linear-history)
- [ ] Require deployments to succeed before merging [info](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches#require-deployments-to-succeed-before-merging)
- [ ] Lock branch
- [ ] Do not allow bypassing the above settings
- [x] Restrict who can push to matching branches
  - [x] Restrict pushes that create matching branches: core maintainers

> [More on the default branch names](https://www.biteinteractive.com/of-git-and-github-master-and-main/).

## Issues

- Every major repo should have issue templates at `.github/ISSUE_TEMPLATE`.
- Issue names should be capitalized, without sentence termination punctuation marks.

> [More on the issue templates](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/about-issue-and-pull-request-templates)

## Commits

- Subject (top) line is up to 50 characters.
  - Should continue the sentence: "This commit will ...".
  - Shouldn't end with a period.
  - Must start with a ["verb"](#commit-verbs) or "type" or "tag".
  - May mark the programming language in broad projects.
- Description lines are optional and limited to 72 characters.
  - Use the body to explain what and why vs. how. Code already answers the latter.
  - If relevant, reference the issue at the end, using the hashtag notation.
- If a commit can't be described with a single verb, it should be split into parts.

The template would be:

```txt
<type>: <description>

[optional body]

[optional footer(s)]
```

An example would be:

```txt
Fix: Short (50 chars or less) summary

More detailed explanatory text. Wrap it to 72 characters. The blank
line separating the summary from the body is critical (unless you omit
the body entirely).

Write your commit message in the imperative: "Fix bug" and not "Fixed
bug" or "Fixes bug." This convention matches up with commit messages
generated by commands like git merge and git revert.

Further paragraphs come after blank lines.

- Bullet points are okay, too.
- Typically a hyphen or asterisk is used for the bullet, followed by a
  single space. Use a hanging indent.
```

### Commit Verbs

We agree on a short list of leading active verbs for the subject line, similar to the [ESLint convention](https://github.com/conventional-changelog/conventional-changelog/tree/master/packages/conventional-changelog-eslint). 

| Verbs    |  Part   | When?                                |
| :------- | :-----: | :----------------------------------- |
| Refactor |    -    | Non-user facing changes              |
| Test     |    -    | Updates **only** to tests            |
| Docs     |    -    | Updates **only** to documentation    |
| Build    |    -    | Compilation fixes                    |
|          |         |                                      |
| Make     | `patch` | New compilation settings, or configs |
| Fix      | `patch` | Bug fixes                            |
| Improve  | `patch` | Small enhancement                    |
| Upgrade  | `patch` | Updating a dependency version        |
|          |         |                                      |
| Add      | `minor` | New feature                          |
|          |         |                                      |
| Breaking | `major` | Backwards-incompatible update        |

Which is a well known, widely adopted set, and self-explanatory set.

###  Why Use Conventional Commits?

- Automatically generating CHANGELOGs.
- Automatically determining a semantic version bump (based on the types of commits landed).
- Communicating the nature of changes to teammates, the public, and other stakeholders.
- Triggering build and publish processes.
- Making it easier for people to contribute to your projects, by allowing them to explore a more structured commit history.


## Automated Semantic Versioning

- Every repo has a root level `VERSION` file.
- Contents are `major.minor.patch` followed by newline.
- CI tools will automatically generate a `CHANGELOG.md` and release notes.

We use a variation of **[Semantic Versioning](https://semver.org/)**.

> [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.

## Libraries and Dependencies

- Check that your [licenses](#licensing) are compatible.
- Prefer dependencies, that are actively maintained, tested, and documented.
- Prefer projects using the same build system.

Choosing dependencies is much harder than it seems.
Those rules can be skipped or bended.
For example, many teams may not consider using projects with less than 1'000 stars on GitHub.
Yet some of the [cleanest](https://github.com/pierresegonne/funky-attractors) and coolest repositories I have seen, have exactly **zero** stars.
Being cautious is a good baselines, but if you want to be competitive, you have to take risks.

## Licensing

- Permissive: MIT, BSD, **Apache** = "don't sue the author".
- "Copyleft" licenses: GPL, AGPL, LGPL = "preserve authorship attribution".
- Public Domain, CC0, Unlicense = "do anything".
- Creative Commons = for hardware guys.

We prefer **[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)**.
That is not compatible with the GPL licenses.
So unless a GPL-licensed project is an **optional** add-on, our projects can't include it.

> [A dev’s guide to open source software licensing](https://github.com/readme/guides/open-source-licensing) by GitHub.
> [Software Licenses in Plain English](https://www.tldrlegal.com/).

## Code Style

Following points are true for all programming and markup languages:

- No trailing whitespaces.
- Every file must end with newline.
- Avoid short variable names in big functions.
- Self-documenting code is better than excessive commenting.
- Use automatic formatters, like `clang-format` for C++ and `black` for Python.

Commenting is a broad topic and everyone has their own style, but here are some good practices and anti-patterns.

### Self Explanatory Comments

The most common mistake novice developers make is using too much boilerplate text, repeating what the code already says.

```c
long f(int n){if(n==0)return 1;else return n*f(n-1);}
int main(){int n;scanf("%d",&n);printf("%ld",f(n));return 0;}
```
```c
// Function to compute factorial
long f(int n) {
    // If n equals zero
    if (n == 0)
        // Return 1 because factorial of 0 is 1
        return 1;
    else
        // Else return n multiplied by factorial of (n-1)
        return n * f(n - 1);
}

// Main function
int main() {
    int n; // Variable to store user input
    scanf("%d", &n); // Read integer from user
    printf("%ld", f(n)); // Print factorial of n
    return 0; // Return 0 to indicate successful execution
}
```
```c
// This function calculates the factorial of a number
long factorial(int number) {
    if (number == 0)
        return 1; // Return 1 if number is zero
    else
        return number * factorial(number - 1); // Recursive call
}

// The main function where execution starts
int main() {
    int input; // Variable to hold user input
    scanf("%d", &input); // Read input from user
    printf("%ld", factorial(input)); // Output the factorial
    return 0; // Successful execution
}
```
```c
long factorial(int number) {
    if (number == 0) return 1;
    return number * factorial(number - 1);
}
int main() {
    int input;
    scanf("%d", &input);
    printf("%ld", factorial(input));
    return 0;
}
```

Fewer words is better.

### Dead Code

Oftentimes you may encounter large sections of code temporarily disabled, commented out.
There is a common and a clean way to do that:

```c
//    start_of_very_long_code_line_
//that_was_wrapped_after_commenting();
//    next_function_call();
```

Alternatively, you can use the preprocessor:

```c
#if 0
    start_of_very_long_code_line_that_was_wrapped_after_commenting();
    next_function_call();
#endif
```

The benefit of the later approach, it doesn't screw up your formatting, doesn't affect the Git history of the block, and is easy to turn back on by flipping the zero.
It also makes searching for dead code sections much easier.

### Color Coding

Several commenting conventions exist, using different line prefixes:

- `#!` in Python or `//!` in C/C++ for important comments.
- `#?` in Python or `//?` in C/C++ for future considerations.
- `#` in Python or `//` in in C/C++ for general inline comments.
- `/** */` in C/C++ for symbol doc-strings.
- `/* */` in C/C++ for multi-line block comments.
- `TODO:` line prefixes for tasks.
