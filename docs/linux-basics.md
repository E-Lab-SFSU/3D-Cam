---
layout: default
title: Linux/Raspberry Pi Script Basics
permalink: /linux-basics/
---

# Linux/Raspberry Pi Script Basics

This document explains the basics of running scripts on Linux/Raspberry Pi.

## Quick Answers

### Why can't I just type `visualize_3d.sh`?

You need `./` before the script name:
```bash
./visualize_3d.sh
```

**Reason:** Linux requires `./` to explicitly tell the system "run this script from the current directory". This is a security feature - it prevents accidentally running scripts if someone puts a malicious file in your current directory.

Without `./`, Linux looks for the script in your PATH (system directories like `/usr/bin`), not in the current folder.

### What does `chmod +x` mean?

`chmod +x` makes a file **executable** (able to be run as a program).

**What it means:**
- `chmod` = "change mode" (file permissions)
- `+x` = "add execute permission"

**Why it's needed:** 
- Linux doesn't automatically make files executable for security
- New files are created without execute permission
- You must explicitly give permission to run scripts

**When to use it:**
- Only needed **once** per file (the first time)
- After that, you can run the script anytime with `./script.sh`

### Why `*.sh` instead of listing files?

The `*` is a **wildcard** (pattern matcher) that matches all files ending with `.sh`.

**What `*.sh` means:**
- `*` = matches any filename
- `.sh` = ends with ".sh"

So `chmod +x *.sh` makes ALL these files executable at once:
- `visualize_3d.sh`
- `detect_pairs.sh`
- `capture_raspi.sh`
- `setup_venv.sh`
- etc.

**Why use it?**
Instead of typing:
```bash
chmod +x visualize_3d.sh
chmod +x detect_pairs.sh
chmod +x setup_venv.sh
# ... (one line for each file)
```

You can do them all at once:
```bash
chmod +x *.sh
```

## Step-by-Step Example

### First Time Setup

1. **Make scripts executable** (one-time setup):
   ```bash
   chmod +x *.sh
   ```
   This makes all `.sh` files executable (including all run scripts and setup_venv.sh).

2. **Run a program:**
   ```bash
   ./visualize_3d.sh
   ```
   Note the `./` before the script name.

### Every Time After That

Just run the script directly:
```bash
./visualize_3d.sh
```

You don't need to run `chmod +x` again - the permission stays set.

## Alternative: Run Without `./`

If you really want to type just `visualize_3d.sh` (without `./`), you can:

1. **Add current directory to PATH** (not recommended for security):
   ```bash
   export PATH=$PATH:.
   visualize_3d.sh  # Now works without ./
   ```

2. **Or use the full path** (works from anywhere):
   ```bash
   /home/user/GitHub/3D-Cam/visualize_3d.sh
   ```

But `./visualize_3d.sh` is the standard, safe way to do it.

## Summary

- **`./`** = required to run scripts in current directory
- **`chmod +x`** = needed once to make files executable
- **`*`** = wildcard to match multiple files
- **Only needed once** = after `chmod +x`, you can run scripts anytime

