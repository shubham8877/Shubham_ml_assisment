# GitHub Deployment Guide
## Step-by-Step: From Local Folder → Public GitHub Link

---

## Prerequisites Checklist

Before starting, make sure you have:
- [ ] VS Code installed → https://code.visualstudio.com/
- [ ] Git installed → https://git-scm.com/downloads
- [ ] A GitHub account → https://github.com/signup
- [ ] This `ml-assessments` folder downloaded on your computer

---

## PART 1 — Create the GitHub Repository

1. Open https://github.com and log in
2. Click the **"+"** icon (top right) → **"New repository"**
3. Fill in:
   - **Repository name:** `ml-engineer-assessment`
   - **Visibility:** Private *(you'll share just the link)*
   - **Initialize with README:** Leave this **unchecked**
4. Click **"Create repository"**
5. Copy the URL shown on the next page — it looks like:
   ```
   https://github.com/YOUR_USERNAME/ml-engineer-assessment.git
   ```

---

## PART 2 — Open the Project in VS Code

1. Open VS Code
2. Go to **File → Open Folder**
3. Navigate to and select the `ml-assessments` folder
4. Click **"Select Folder"**

You should see this structure in the Explorer panel on the left:
```
ml-assessments/
├── .gitignore
├── .env.example
├── README.md
├── assessment1_anomaly_detection/
└── assessment2_document_summarization/
```

---

## PART 3 — Open the Terminal in VS Code

Press **Ctrl + ` ** (backtick) on Windows/Linux
or **Cmd + ` ** on Mac

This opens the integrated terminal at the bottom of VS Code.

---

## PART 4 — Configure Git (First Time Only)

If you've never used Git before, set your identity:

```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@gmail.com"
```

Verify Git is installed:
```bash
git --version
# Should print: git version 2.x.x
```

---

## PART 5 — Initialize and Push to GitHub

Run these commands **one by one** in the VS Code terminal:

```bash
# Step 1: Initialize a git repository in this folder
git init

# Step 2: Stage ALL files for the first commit
git add .

# Step 3: Check what will be committed (optional but recommended)
git status

# Step 4: Create the first commit
git commit -m "Add Assessment 1 (Anomaly Detection) and Assessment 2 (Document Summarization)"

# Step 5: Connect your local folder to the GitHub repository
# REPLACE "YOUR_USERNAME" with your actual GitHub username
git remote add origin https://github.com/YOUR_USERNAME/ml-engineer-assessment.git

# Step 6: Rename the default branch to "main" (GitHub standard)
git branch -M main

# Step 7: Push your code to GitHub
git push -u origin main
```

When prompted, enter your GitHub username and password.

> **Note:** GitHub no longer accepts passwords for pushing. If you get an error,
> you need a Personal Access Token. See the Troubleshooting section below.

---

## PART 6 — Verify on GitHub

1. Go to `https://github.com/YOUR_USERNAME/ml-engineer-assessment`
2. You should see all your files listed with the README rendered at the bottom
3. Check that sensitive files are NOT present:
   - ✅ `.env.example` should be there
   - ❌ `.env` should NOT be there (protected by .gitignore)
   - ❌ `models/*.pkl` should NOT be there

---

## PART 7 — Share the Link

The URL to send is simply:
```
https://github.com/YOUR_USERNAME/ml-engineer-assessment
```

If you set it to Private, the recruiter needs access:
- Go to **Settings → Collaborators → Add people**
- Enter their GitHub username or email

---

## PART 8 — Making Changes After Initial Push

If you edit any file later:

```bash
# Stage the changed files
git add .

# Commit with a descriptive message
git commit -m "Fix: improve threshold tuning logic in train.py"

# Push to GitHub
git push
```

---

## Troubleshooting

### "Authentication failed" when pushing

GitHub removed password auth. Create a token:
1. GitHub → **Settings → Developer Settings → Personal Access Tokens → Tokens (classic)**
2. Click **Generate new token (classic)**
3. Check the **"repo"** scope
4. Copy the token (you only see it once)
5. Use the token as your password when Git asks

### "fatal: remote origin already exists"

```bash
git remote set-url origin https://github.com/YOUR_USERNAME/ml-engineer-assessment.git
```

### Files not showing up on GitHub

```bash
git status   # Check what's staged
git add .    # Re-stage everything
git push     # Push again
```

### Large files rejected (>100MB)

The `.gitignore` already excludes model binaries (.pkl, .pt).
If a large file slipped through:
```bash
git rm --cached path/to/large/file
git commit -m "Remove large file from tracking"
git push
```

---

## Quick Reference Commands

| Action | Command |
|--------|---------|
| Check status | `git status` |
| Stage all files | `git add .` |
| Commit changes | `git commit -m "your message"` |
| Push to GitHub | `git push` |
| See commit history | `git log --oneline` |
| Undo last commit (keep files) | `git reset HEAD~1` |
