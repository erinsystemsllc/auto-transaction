const getBranchName = (branch) =>
  branch ? /^(?:refs\/heads\/)?(?<branch>.+)$/i.exec(branch)[1] : undefined;

const branchName = getBranchName(process.env.GITHUB_REF);

module.exports = {
  branches: [
    'main',
    {
      name: branchName,
      /* eslint-disable no-template-curly-in-string */
      prerelease: '${name.replace(/\\//g, "-")}',
    },
  ],
  /* eslint-disable no-template-curly-in-string */
  tagFormat: '${version}',
  plugins: [
    [
      '@semantic-release/commit-analyzer',
      {
        releaseRules: [
          { type: 'refactor', release: 'patch' },
          { type: 'refactor', scope: 'core-*', release: 'minor' },
        ],
      },
    ],
    '@semantic-release/release-notes-generator',
    ...(branchName === 'main'
      ? [
          '@semantic-release/changelog',
          [
            '@semantic-release/npm',
            {
              npmPublish: false,
            },
          ],
        ]
      : []),
    [
      '@semantic-release/exec',
      {
        prepareCmd:
          /* eslint-disable no-template-curly-in-string */
          './cicd/prepare-docker.sh ${nextRelease.version}',
        /* eslint-disable no-template-curly-in-string */
        publishCmd: './cicd/publish-docker.sh ${nextRelease.version}',
      },
    ],
    '@semantic-release/github',
    [
      '@semantic-release/git',
      {
        assets: [
          'CHANGELOG.md',
          'package.json',
          'package-lock.json',
          'npm-shrinkwrap.json',
          'charts/express-api/Chart.yaml',
        ],
        /* eslint-disable no-template-curly-in-string */
        message:
          'chore: release 🚀 ${nextRelease.version} [skip ci]\n\n${nextRelease.notes}',
      },
    ],
  ],
};
