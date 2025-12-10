// Sample TypeScript file for testing
interface User {
    id: number;
    name: string;
    email: string;
}

interface Product {
    id: number;
    name: string;
    price: number;
    inStock: boolean;
}

class UserService {
    private users: User[] = [];

    addUser(user: User): void {
        this.users.push(user);
    }

    findUserById(id: number): User | undefined {
        return this.users.find(u => u.id === id);
    }

    getAllUsers(): User[] {
        return [...this.users];
    }

    deleteUser(id: number): boolean {
        const index = this.users.findIndex(u => u.id === id);
        if (index !== -1) {
            this.users.splice(index, 1);
            return true;
        }
        return false;
    }
}

type OrderStatus = 'pending' | 'processing' | 'shipped' | 'delivered';

interface Order {
    id: string;
    userId: number;
    products: Product[];
    status: OrderStatus;
    totalAmount: number;
}

function calculateOrderTotal(products: Product[]): number {
    return products.reduce((sum, product) => sum + product.price, 0);
}

export { User, Product, UserService, Order, OrderStatus, calculateOrderTotal };
