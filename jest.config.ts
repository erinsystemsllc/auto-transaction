import type { Config } from '@jest/types';

const config: Config.InitialOptions = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  moduleDirectories: ['node_modules', 'src'],
  testTimeout: 60000,
  roots: ['test'],
  collectCoverage: true,
  coverageProvider: 'v8',
  coverageDirectory: '../coverage',
  coveragePathIgnorePatterns: ['/node_modules/', '/__test__/'],
  coverageThreshold: {
    // Based on https://testing.googleblog.com/2020/08/code-coverage-best-practices.html
    global: {
      branches: 60,
      functions: 60,
      lines: 60,
    },
  },
  coverageReporters: ['json', 'json-summary', 'text'],
};
export default config;
