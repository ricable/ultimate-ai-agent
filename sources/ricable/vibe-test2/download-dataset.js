const { downloadFile } = require('@huggingface/hub');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

const DATASET_REPO = 'otellm/3gpp_knowledgeGraph';
const DATA_DIR = path.join(__dirname, 'data');

async function downloadDataset() {
  try {
    // Create data directory if it doesn't exist
    if (!fs.existsSync(DATA_DIR)) {
      fs.mkdirSync(DATA_DIR, { recursive: true });
      console.log('Created data directory');
    }

    console.log(`Downloading dataset from ${DATASET_REPO}...`);

    // Download the GraphML file from the Hugging Face dataset
    // The actual filename might vary - common names are 3gpp.graphml, knowledge_graph.graphml, etc.
    const possibleFilenames = [
      '3gpp.graphml',
      'knowledge_graph.graphml',
      'graph.graphml',
      'data.graphml'
    ];

    const credentials = process.env.HF_TOKEN ? { accessToken: process.env.HF_TOKEN } : {};

    for (const filename of possibleFilenames) {
      try {
        const outputPath = path.join(DATA_DIR, filename);

        await downloadFile({
          repo: DATASET_REPO,
          path: filename,
          ...credentials
        }).then(response => {
          if (response) {
            const fileStream = fs.createWriteStream(outputPath);
            response.body.pipe(fileStream);

            return new Promise((resolve, reject) => {
              fileStream.on('finish', () => {
                console.log(`✓ Successfully downloaded: ${filename}`);
                resolve();
              });
              fileStream.on('error', reject);
            });
          }
        });

        break; // If successful, stop trying other filenames
      } catch (err) {
        console.log(`File ${filename} not found, trying next...`);
      }
    }

    console.log('Dataset download complete!');
    console.log(`Files saved to: ${DATA_DIR}`);

  } catch (error) {
    console.error('Error downloading dataset:', error.message);
    console.error('\nTroubleshooting:');
    console.error('1. Check if the dataset repository exists: https://huggingface.co/datasets/' + DATASET_REPO);
    console.error('2. If the dataset is private, make sure HF_TOKEN is set in .env');
    console.error('3. Verify the GraphML file name in the dataset repository');
    process.exit(1);
  }
}

// Run if called directly
if (require.main === module) {
  downloadDataset();
}

module.exports = { downloadDataset };
