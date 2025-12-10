describe('Example tests', () => {
  test('should pass basic test', () => {
    expect(1 + 1).toBe(2);
  });
  
  test('should handle strings', () => {
    expect('Hello, World!').toContain('Hello');
  });
});