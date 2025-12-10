// Sample JavaScript file for testing
function calculateSum(a, b) {
    return a + b;
}

function calculateProduct(a, b) {
    return a * b;
}

class MathOperations {
    constructor() {
        this.history = [];
    }

    add(a, b) {
        const result = a + b;
        this.history.push({ op: 'add', result });
        return result;
    }

    multiply(a, b) {
        const result = a * b;
        this.history.push({ op: 'multiply', result });
        return result;
    }

    getHistory() {
        return this.history;
    }
}

const utils = {
    isEven: (n) => n % 2 === 0,
    isOdd: (n) => n % 2 !== 0,
    square: (n) => n * n
};

export { calculateSum, calculateProduct, MathOperations, utils };
