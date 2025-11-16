# GitHub Pages Setup Instructions

This repository now includes a Jekyll documentation site that can be hosted on GitHub Pages at `https://e-lab-sfsu.github.io/3D-Cam`.

## Quick Setup

1. **Enable GitHub Pages:**
   - Go to your repository on GitHub: `https://github.com/e-lab-sfsu/3D-Cam`
   - Click on **Settings** (top menu)
   - Scroll down to **Pages** in the left sidebar
   - Under **Source**, select:
     - **Branch**: `main`
     - **Folder**: `/docs`
   - Click **Save**

2. **Wait for deployment:**
   - GitHub will automatically build and deploy your site
   - This usually takes 1-2 minutes
   - You can check the status in the **Actions** tab

3. **Access your site:**
   - Your site will be available at: `https://e-lab-sfsu.github.io/3D-Cam`
   - It may take a few minutes for the DNS to propagate

## What's Included

The documentation site includes:
- **Homepage** (`index.md`) - Project overview and quick start
- **Usage Guide** (`usage.md`) - Complete workflow instructions
- **Setup Instructions** (`setup.md`) - Installation and configuration
- **Scripts Reference** (`scripts.md`) - All available scripts
- **Linux Basics** (`linux-basics.md`) - Guide for Linux/Raspberry Pi users

## Local Development

To test the site locally before pushing:

1. **Install Ruby and Bundler** (if not already installed):
   ```bash
   # On macOS (with Homebrew)
   brew install ruby
   
   # On Ubuntu/Debian
   sudo apt-get install ruby-full build-essential zlib1g-dev
   
   # On Windows
   # Download RubyInstaller from https://rubyinstaller.org/
   ```

2. **Install Jekyll and dependencies:**
   ```bash
   cd docs
   bundle install
   ```

3. **Run the local server:**
   ```bash
   bundle exec jekyll serve
   ```

4. **View the site:**
   - Open `http://localhost:4000/3D-Cam` in your browser
   - The site will auto-reload when you make changes

## Making Changes

1. **Edit documentation files:**
   - All documentation is in the `docs/` folder
   - Files are in Markdown format (`.md`)
   - Edit them like any other text file

2. **Add new pages:**
   - Create a new `.md` file in `docs/`
   - Add front matter at the top:
     ```yaml
     ---
     layout: default
     title: Your Page Title
     permalink: /your-page-url/
     ---
     ```
   - Add to navigation in `docs/_config.yml` under `header_pages` if desired

3. **Commit and push:**
   ```bash
   git add docs/
   git commit -m "Update documentation"
   git push
   ```
   - GitHub Actions will automatically rebuild and deploy the site

## Troubleshooting

### Site not updating
- Check the **Actions** tab for build errors
- Ensure you're pushing to the `main` branch
- Wait a few minutes for deployment to complete

### Build errors
- Check the Actions tab for error messages
- Ensure `docs/Gemfile` is present
- Try running `bundle install` locally to check for dependency issues

### 404 errors
- Verify the `baseurl` in `docs/_config.yml` is set to `/3D-Cam`
- Check that permalinks in page front matter are correct

## Files Structure

```
docs/
├── _config.yml          # Jekyll configuration
├── index.md             # Homepage
├── usage.md             # Usage guide
├── setup.md             # Setup instructions
├── scripts.md           # Scripts reference
├── linux-basics.md      # Linux basics guide
├── Gemfile              # Ruby dependencies
└── README.md            # Local development instructions

.github/
└── workflows/
    └── pages.yml        # GitHub Actions workflow for building/deploying
```

## Need Help?

- [Jekyll Documentation](https://jekyllrb.com/docs/)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [Minima Theme Documentation](https://github.com/jekyll/minima)

