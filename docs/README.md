# Documentation Site

This directory contains the Jekyll documentation site for the 3D-Cam project.

## Local Development

To test the site locally:

1. Install Ruby and Bundler (if not already installed)
2. Install dependencies:
   ```bash
   cd docs
   bundle install
   ```
3. Run Jekyll server:
   ```bash
   bundle exec jekyll serve
   ```
4. Open http://localhost:4000/3D-Cam in your browser

## GitHub Pages Setup

1. Go to your repository settings on GitHub
2. Navigate to "Pages" in the left sidebar
3. Under "Source", select:
   - Branch: `main`
   - Folder: `/docs`
4. Click "Save"
5. The site will be available at: `https://e-lab-sfsu.github.io/3D-Cam`

The GitHub Actions workflow (`.github/workflows/pages.yml`) will automatically build and deploy the site when you push changes to the `docs/` folder.

## Adding New Pages

1. Create a new `.md` file in the `docs/` directory
2. Add front matter at the top:
   ```yaml
   ---
   layout: default
   title: Your Page Title
   permalink: /your-page-url/
   ---
   ```
3. Add the page to navigation in `_config.yml` under `header_pages` if desired

