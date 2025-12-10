let wasm;
export function __wbg_set_wasm(val) {
    wasm = val;
}


let cachedUint8ArrayMemory0 = null;

function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });

cachedTextDecoder.decode();

const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let heap = new Array(128).fill(undefined);

heap.push(undefined, null, true, false);

let heap_next = heap.length;

function addHeapObject(obj) {
    if (heap_next === heap.length) heap.push(heap.length + 1);
    const idx = heap_next;
    heap_next = heap[idx];

    heap[idx] = obj;
    return idx;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        wasm.__wbindgen_export2(addHeapObject(e));
    }
}

function getObject(idx) { return heap[idx]; }

let WASM_VECTOR_LEN = 0;

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    }
}

function passStringToWasm0(arg, malloc, realloc) {

    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }

    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

let cachedDataViewMemory0 = null;

function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

function dropObject(idx) {
    if (idx < 132) return;
    heap[idx] = heap_next;
    heap_next = idx;
}

function takeObject(idx) {
    const ret = getObject(idx);
    dropObject(idx);
    return ret;
}
/**
 * Initialize WASM module
 */
export function wasm_init() {
    wasm.wasm_init();
}

/**
 * Type of jujutsu operation
 *
 * Represents the various operations that can be performed in a jujutsu repository.
 * @enum {0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29}
 */
export const OperationType = Object.freeze({
    /**
     * Create a new commit
     */
    Commit: 0, "0": "Commit",
    /**
     * Snapshot operation (automatic)
     */
    Snapshot: 1, "1": "Snapshot",
    /**
     * Describe/amend commit message
     */
    Describe: 2, "2": "Describe",
    /**
     * New commit creation
     */
    New: 3, "3": "New",
    /**
     * Edit commit
     */
    Edit: 4, "4": "Edit",
    /**
     * Abandon a commit
     */
    Abandon: 5, "5": "Abandon",
    /**
     * Rebase commits
     */
    Rebase: 6, "6": "Rebase",
    /**
     * Squash commits
     */
    Squash: 7, "7": "Squash",
    /**
     * Resolve conflicts
     */
    Resolve: 8, "8": "Resolve",
    /**
     * Branch operation
     */
    Branch: 9, "9": "Branch",
    /**
     * Delete a branch
     */
    BranchDelete: 10, "10": "BranchDelete",
    /**
     * Bookmark operation
     */
    Bookmark: 11, "11": "Bookmark",
    /**
     * Create a tag
     */
    Tag: 12, "12": "Tag",
    /**
     * Checkout a commit
     */
    Checkout: 13, "13": "Checkout",
    /**
     * Restore files
     */
    Restore: 14, "14": "Restore",
    /**
     * Split a commit
     */
    Split: 15, "15": "Split",
    /**
     * Duplicate a commit
     */
    Duplicate: 16, "16": "Duplicate",
    /**
     * Undo last operation
     */
    Undo: 17, "17": "Undo",
    /**
     * Fetch from remote
     */
    Fetch: 18, "18": "Fetch",
    /**
     * Git fetch
     */
    GitFetch: 19, "19": "GitFetch",
    /**
     * Push to remote
     */
    Push: 20, "20": "Push",
    /**
     * Git push
     */
    GitPush: 21, "21": "GitPush",
    /**
     * Clone repository
     */
    Clone: 22, "22": "Clone",
    /**
     * Initialize repository
     */
    Init: 23, "23": "Init",
    /**
     * Git import
     */
    GitImport: 24, "24": "GitImport",
    /**
     * Git export
     */
    GitExport: 25, "25": "GitExport",
    /**
     * Move changes
     */
    Move: 26, "26": "Move",
    /**
     * Diffedit
     */
    Diffedit: 27, "27": "Diffedit",
    /**
     * Merge branches
     */
    Merge: 28, "28": "Merge",
    /**
     * Unknown operation type
     */
    Unknown: 29, "29": "Unknown",
});

const JJBranchFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_jjbranch_free(ptr >>> 0, 1));
/**
 * Branch information
 *
 * Represents a branch in the jujutsu repository.
 *
 * # Examples
 *
 * ```rust
 * use agentic_jujutsu::types::JJBranch;
 *
 * let branch = JJBranch::new("feature/new-api".to_string(), "def456".to_string(), false);
 * assert_eq!(branch.name, "feature/new-api");
 * ```
 */
export class JJBranch {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        JJBranchFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_jjbranch_free(ptr, 0);
    }
    /**
     * Whether this is a remote branch
     * @returns {boolean}
     */
    get is_remote() {
        const ret = wasm.__wbg_get_jjbranch_is_remote(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Whether this is a remote branch
     * @param {boolean} arg0
     */
    set is_remote(arg0) {
        wasm.__wbg_set_jjbranch_is_remote(this.__wbg_ptr, arg0);
    }
    /**
     * Whether this branch is tracking a remote
     * @returns {boolean}
     */
    get is_tracking() {
        const ret = wasm.__wbg_get_jjbranch_is_tracking(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Whether this branch is tracking a remote
     * @param {boolean} arg0
     */
    set is_tracking(arg0) {
        wasm.__wbg_set_jjbranch_is_tracking(this.__wbg_ptr, arg0);
    }
    /**
     * Whether this is the current branch
     * @returns {boolean}
     */
    get is_current() {
        const ret = wasm.__wbg_get_jjbranch_is_current(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Whether this is the current branch
     * @param {boolean} arg0
     */
    set is_current(arg0) {
        wasm.__wbg_set_jjbranch_is_current(this.__wbg_ptr, arg0);
    }
    /**
     * Create a new branch
     * @param {string} name
     * @param {string} target
     * @param {boolean} is_remote
     */
    constructor(name, target, is_remote) {
        const ptr0 = passStringToWasm0(name, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(target, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.jjbranch_new(ptr0, len0, ptr1, len1, is_remote);
        this.__wbg_ptr = ret >>> 0;
        JJBranchFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Get name (for WASM)
     * @returns {string}
     */
    get name() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjbranch_name(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Get target (for WASM)
     * @returns {string}
     */
    get target() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjbranch_target(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Get remote (for WASM)
     * @returns {string | undefined}
     */
    get remote() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjbranch_remote(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            let v1;
            if (r0 !== 0) {
                v1 = getStringFromWasm0(r0, r1).slice();
                wasm.__wbindgen_export(r0, r1 * 1, 1);
            }
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Set remote name
     * @param {string} remote
     */
    set_remote(remote) {
        const ptr0 = passStringToWasm0(remote, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
        const len0 = WASM_VECTOR_LEN;
        wasm.jjbranch_set_remote(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Get full branch name (e.g., "origin/main")
     * @returns {string}
     */
    full_name() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjbranch_full_name(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Short target ID (first 12 characters)
     * @returns {string}
     */
    short_target() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjbranch_short_target(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Get creation timestamp as ISO 8601 string (for WASM)
     * @returns {string}
     */
    get created_at_iso() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjbranch_created_at_iso(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
}
if (Symbol.dispose) JJBranch.prototype[Symbol.dispose] = JJBranch.prototype.free;

const JJCommitFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_jjcommit_free(ptr >>> 0, 1));
/**
 * Commit metadata
 *
 * Represents a single commit in the jujutsu repository with all associated metadata.
 *
 * # Examples
 *
 * ```rust
 * use agentic_jujutsu::types::JJCommit;
 *
 * let commit = JJCommit::builder()
 *     .id("abc123".to_string())
 *     .message("Add new feature".to_string())
 *     .author("Bob".to_string())
 *     .build();
 *
 * assert_eq!(commit.message, "Add new feature");
 * ```
 */
export class JJCommit {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        JJCommitFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_jjcommit_free(ptr, 0);
    }
    /**
     * Whether this is a merge commit
     * @returns {boolean}
     */
    get is_merge() {
        const ret = wasm.__wbg_get_jjcommit_is_merge(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Whether this is a merge commit
     * @param {boolean} arg0
     */
    set is_merge(arg0) {
        wasm.__wbg_set_jjcommit_is_merge(this.__wbg_ptr, arg0);
    }
    /**
     * Whether this commit has conflicts
     * @returns {boolean}
     */
    get has_conflicts() {
        const ret = wasm.__wbg_get_jjcommit_has_conflicts(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Whether this commit has conflicts
     * @param {boolean} arg0
     */
    set has_conflicts(arg0) {
        wasm.__wbg_set_jjcommit_has_conflicts(this.__wbg_ptr, arg0);
    }
    /**
     * Whether this is an empty commit
     * @returns {boolean}
     */
    get is_empty() {
        const ret = wasm.__wbg_get_jjcommit_is_empty(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Whether this is an empty commit
     * @param {boolean} arg0
     */
    set is_empty(arg0) {
        wasm.__wbg_set_jjcommit_is_empty(this.__wbg_ptr, arg0);
    }
    /**
     * Create a new commit with minimal fields
     * @param {string} id
     * @param {string} change_id
     * @param {string} message
     * @param {string} author
     * @param {string} author_email
     */
    constructor(id, change_id, message, author, author_email) {
        const ptr0 = passStringToWasm0(id, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(change_id, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(message, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
        const len2 = WASM_VECTOR_LEN;
        const ptr3 = passStringToWasm0(author, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
        const len3 = WASM_VECTOR_LEN;
        const ptr4 = passStringToWasm0(author_email, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
        const len4 = WASM_VECTOR_LEN;
        const ret = wasm.jjcommit_new(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, ptr4, len4);
        this.__wbg_ptr = ret >>> 0;
        JJCommitFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Get ID (for WASM)
     * @returns {string}
     */
    get id() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjcommit_id(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Get change ID (for WASM)
     * @returns {string}
     */
    get change_id() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjcommit_change_id(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Get message (for WASM)
     * @returns {string}
     */
    get message() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjcommit_message(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Get author (for WASM)
     * @returns {string}
     */
    get author() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjcommit_author(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Get author email (for WASM)
     * @returns {string}
     */
    get author_email() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjcommit_author_email(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Get timestamp as ISO 8601 string (for WASM)
     * @returns {string}
     */
    timestamp_iso() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjcommit_timestamp_iso(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Get parent count
     * @returns {number}
     */
    parent_count() {
        const ret = wasm.jjcommit_parent_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Short commit ID (first 12 characters)
     * @returns {string}
     */
    short_id() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjcommit_short_id(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Short change ID (first 12 characters)
     * @returns {string}
     */
    short_change_id() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjcommit_short_change_id(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Get parents as JSON string (for WASM)
     * @returns {string}
     */
    get parents_json() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjcommit_parents_json(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Get branches as JSON string (for WASM)
     * @returns {string}
     */
    get branches_json() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjcommit_branches_json(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
}
if (Symbol.dispose) JJCommit.prototype[Symbol.dispose] = JJCommit.prototype.free;

const JJConfigFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_jjconfig_free(ptr >>> 0, 1));
/**
 * Configuration for JJWrapper
 */
export class JJConfig {

    static __wrap(ptr) {
        ptr = ptr >>> 0;
        const obj = Object.create(JJConfig.prototype);
        obj.__wbg_ptr = ptr;
        JJConfigFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        JJConfigFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_jjconfig_free(ptr, 0);
    }
    /**
     * Timeout for operations in milliseconds
     * @returns {bigint}
     */
    get timeout_ms() {
        const ret = wasm.__wbg_get_jjconfig_timeout_ms(this.__wbg_ptr);
        return BigInt.asUintN(64, ret);
    }
    /**
     * Timeout for operations in milliseconds
     * @param {bigint} arg0
     */
    set timeout_ms(arg0) {
        wasm.__wbg_set_jjconfig_timeout_ms(this.__wbg_ptr, arg0);
    }
    /**
     * Enable verbose logging
     * @returns {boolean}
     */
    get verbose() {
        const ret = wasm.__wbg_get_jjconfig_verbose(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Enable verbose logging
     * @param {boolean} arg0
     */
    set verbose(arg0) {
        wasm.__wbg_set_jjconfig_verbose(this.__wbg_ptr, arg0);
    }
    /**
     * Maximum operation log entries to keep in memory
     * @returns {number}
     */
    get max_log_entries() {
        const ret = wasm.__wbg_get_jjconfig_max_log_entries(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Maximum operation log entries to keep in memory
     * @param {number} arg0
     */
    set max_log_entries(arg0) {
        wasm.__wbg_set_jjconfig_max_log_entries(this.__wbg_ptr, arg0);
    }
    /**
     * Enable AgentDB sync
     * @returns {boolean}
     */
    get enable_agentdb_sync() {
        const ret = wasm.__wbg_get_jjconfig_enable_agentdb_sync(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Enable AgentDB sync
     * @param {boolean} arg0
     */
    set enable_agentdb_sync(arg0) {
        wasm.__wbg_set_jjconfig_enable_agentdb_sync(this.__wbg_ptr, arg0);
    }
    /**
     * Create new configuration with defaults
     */
    constructor() {
        const ret = wasm.jjconfig_default_config();
        this.__wbg_ptr = ret >>> 0;
        JJConfigFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Create default configuration
     * @returns {JJConfig}
     */
    static default_config() {
        const ret = wasm.jjconfig_default_config();
        return JJConfig.__wrap(ret);
    }
    /**
     * Get jj executable path
     * @returns {string}
     */
    get jj_path() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjconfig_jj_path(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Set jj executable path
     * @param {string} path
     */
    set jj_path(path) {
        const ptr0 = passStringToWasm0(path, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
        const len0 = WASM_VECTOR_LEN;
        wasm.jjconfig_set_jj_path(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Get repository path
     * @returns {string}
     */
    get repo_path() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjconfig_repo_path(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Set repository path
     * @param {string} path
     */
    set repo_path(path) {
        const ptr0 = passStringToWasm0(path, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
        const len0 = WASM_VECTOR_LEN;
        wasm.jjconfig_set_repo_path(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Set jj executable path (builder pattern)
     * @param {string} path
     * @returns {JJConfig}
     */
    with_jj_path(path) {
        const ptr = this.__destroy_into_raw();
        const ptr0 = passStringToWasm0(path, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.jjconfig_with_jj_path(ptr, ptr0, len0);
        return JJConfig.__wrap(ret);
    }
    /**
     * Set repository path (builder pattern)
     * @param {string} path
     * @returns {JJConfig}
     */
    with_repo_path(path) {
        const ptr = this.__destroy_into_raw();
        const ptr0 = passStringToWasm0(path, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
        const len0 = WASM_VECTOR_LEN;
        const ret = wasm.jjconfig_with_repo_path(ptr, ptr0, len0);
        return JJConfig.__wrap(ret);
    }
    /**
     * Set operation timeout
     * @param {bigint} timeout_ms
     * @returns {JJConfig}
     */
    with_timeout(timeout_ms) {
        const ptr = this.__destroy_into_raw();
        const ret = wasm.jjconfig_with_timeout(ptr, timeout_ms);
        return JJConfig.__wrap(ret);
    }
    /**
     * Enable verbose logging
     * @param {boolean} verbose
     * @returns {JJConfig}
     */
    with_verbose(verbose) {
        const ptr = this.__destroy_into_raw();
        const ret = wasm.jjconfig_with_verbose(ptr, verbose);
        return JJConfig.__wrap(ret);
    }
    /**
     * Set max log entries
     * @param {number} max
     * @returns {JJConfig}
     */
    with_max_log_entries(max) {
        const ptr = this.__destroy_into_raw();
        const ret = wasm.jjconfig_with_max_log_entries(ptr, max);
        return JJConfig.__wrap(ret);
    }
    /**
     * Enable AgentDB synchronization
     * @param {boolean} enable
     * @returns {JJConfig}
     */
    with_agentdb_sync(enable) {
        const ptr = this.__destroy_into_raw();
        const ret = wasm.jjconfig_with_agentdb_sync(ptr, enable);
        return JJConfig.__wrap(ret);
    }
}
if (Symbol.dispose) JJConfig.prototype[Symbol.dispose] = JJConfig.prototype.free;

const JJConflictFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_jjconflict_free(ptr >>> 0, 1));
/**
 * Conflict representation
 *
 * Represents a merge conflict with detailed information about conflicting sides.
 *
 * # Examples
 *
 * ```rust
 * use agentic_jujutsu::types::JJConflict;
 *
 * let conflict = JJConflict::builder()
 *     .path("src/main.rs".to_string())
 *     .num_conflicts(1)
 *     .conflict_type("content".to_string())
 *     .build();
 * ```
 */
export class JJConflict {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        JJConflictFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_jjconflict_free(ptr, 0);
    }
    /**
     * Number of conflict markers
     * @returns {number}
     */
    get num_conflicts() {
        const ret = wasm.__wbg_get_jjconflict_num_conflicts(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Number of conflict markers
     * @param {number} arg0
     */
    set num_conflicts(arg0) {
        wasm.__wbg_set_jjconflict_num_conflicts(this.__wbg_ptr, arg0);
    }
    /**
     * Whether conflict is binary (non-text)
     * @returns {boolean}
     */
    get is_binary() {
        const ret = wasm.__wbg_get_jjconflict_is_binary(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Whether conflict is binary (non-text)
     * @param {boolean} arg0
     */
    set is_binary(arg0) {
        wasm.__wbg_set_jjconflict_is_binary(this.__wbg_ptr, arg0);
    }
    /**
     * Whether conflict is resolved
     * @returns {boolean}
     */
    get is_resolved() {
        const ret = wasm.__wbg_get_jjconflict_is_resolved(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Whether conflict is resolved
     * @param {boolean} arg0
     */
    set is_resolved(arg0) {
        wasm.__wbg_set_jjconflict_is_resolved(this.__wbg_ptr, arg0);
    }
    /**
     * Create a new conflict
     * @param {string} path
     * @param {number} num_conflicts
     * @param {string} conflict_type
     */
    constructor(path, num_conflicts, conflict_type) {
        const ptr0 = passStringToWasm0(path, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(conflict_type, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.jjconflict_new(ptr0, len0, num_conflicts, ptr1, len1);
        this.__wbg_ptr = ret >>> 0;
        JJConflictFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Get ID (for WASM)
     * @returns {string}
     */
    get id() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjconflict_id(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Get path (for WASM)
     * @returns {string}
     */
    get path() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjconflict_path(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Get conflict type (for WASM)
     * @returns {string}
     */
    get conflict_type() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjconflict_conflict_type(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Get resolution strategy (for WASM)
     * @returns {string | undefined}
     */
    get resolution_strategy() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjconflict_resolution_strategy(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            let v1;
            if (r0 !== 0) {
                v1 = getStringFromWasm0(r0, r1).slice();
                wasm.__wbindgen_export(r0, r1 * 1, 1);
            }
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Add a side to the conflict
     * @param {string} side
     */
    add_side(side) {
        const ptr0 = passStringToWasm0(side, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
        const len0 = WASM_VECTOR_LEN;
        wasm.jjconflict_add_side(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Get the number of sides
     * @returns {number}
     */
    num_sides() {
        const ret = wasm.jjconflict_num_sides(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Get sides as JSON string (for WASM)
     * @returns {string}
     */
    get sides_json() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjconflict_sides_json(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
}
if (Symbol.dispose) JJConflict.prototype[Symbol.dispose] = JJConflict.prototype.free;

const JJOperationFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_jjoperation_free(ptr >>> 0, 1));
/**
 * Single jujutsu operation
 *
 * Represents a single operation in the jujutsu operation log with metadata.
 *
 * # Examples
 *
 * ```rust
 * use agentic_jujutsu::operations::{JJOperation, OperationType};
 *
 * let op = JJOperation::builder()
 *     .operation_type(OperationType::Commit)
 *     .command("jj commit".to_string())
 *     .user("alice".to_string())
 *     .build();
 * ```
 */
export class JJOperation {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        JJOperationFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_jjoperation_free(ptr, 0);
    }
    /**
     * Duration in milliseconds
     * @returns {bigint}
     */
    get duration_ms() {
        const ret = wasm.__wbg_get_jjoperation_duration_ms(this.__wbg_ptr);
        return BigInt.asUintN(64, ret);
    }
    /**
     * Duration in milliseconds
     * @param {bigint} arg0
     */
    set duration_ms(arg0) {
        wasm.__wbg_set_jjoperation_duration_ms(this.__wbg_ptr, arg0);
    }
    /**
     * Whether this operation was successful
     * @returns {boolean}
     */
    get success() {
        const ret = wasm.__wbg_get_jjoperation_success(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Whether this operation was successful
     * @param {boolean} arg0
     */
    set success(arg0) {
        wasm.__wbg_set_jjoperation_success(this.__wbg_ptr, arg0);
    }
    /**
     * Create a new operation
     * @param {string} operation_id
     * @param {string} command
     * @param {string} user
     * @param {string} hostname
     */
    constructor(operation_id, command, user, hostname) {
        const ptr0 = passStringToWasm0(operation_id, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(command, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
        const len1 = WASM_VECTOR_LEN;
        const ptr2 = passStringToWasm0(user, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
        const len2 = WASM_VECTOR_LEN;
        const ptr3 = passStringToWasm0(hostname, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
        const len3 = WASM_VECTOR_LEN;
        const ret = wasm.jjoperation_new(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3);
        this.__wbg_ptr = ret >>> 0;
        JJOperationFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Get ID (for WASM)
     * @returns {string}
     */
    get id() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjoperation_id(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Get operation ID (for WASM)
     * @returns {string}
     */
    get operation_id() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjoperation_operation_id(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Get command (for WASM)
     * @returns {string}
     */
    get command() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjoperation_command(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Get user (for WASM)
     * @returns {string}
     */
    get user() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjoperation_user(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Get hostname (for WASM)
     * @returns {string}
     */
    get hostname() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjoperation_hostname(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Get parent ID (for WASM)
     * @returns {string | undefined}
     */
    get parent_id() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjoperation_parent_id(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            let v1;
            if (r0 !== 0) {
                v1 = getStringFromWasm0(r0, r1).slice();
                wasm.__wbindgen_export(r0, r1 * 1, 1);
            }
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Get error (for WASM)
     * @returns {string | undefined}
     */
    get error() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjoperation_error(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            let v1;
            if (r0 !== 0) {
                v1 = getStringFromWasm0(r0, r1).slice();
                wasm.__wbindgen_export(r0, r1 * 1, 1);
            }
            return v1;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Get operation type as string (for WASM)
     * @returns {string}
     */
    get operation_type_str() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjoperation_operation_type_str(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Set operation type from string
     * @param {string} type_str
     */
    set_operation_type(type_str) {
        const ptr0 = passStringToWasm0(type_str, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
        const len0 = WASM_VECTOR_LEN;
        wasm.jjoperation_set_operation_type(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Get timestamp as ISO 8601 string (for WASM)
     * @returns {string}
     */
    timestamp_iso() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjoperation_timestamp_iso(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Short operation ID (first 12 characters)
     * @returns {string}
     */
    short_id() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjoperation_short_id(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Check if operation is a snapshot
     * @returns {boolean}
     */
    is_snapshot() {
        const ret = wasm.jjoperation_is_snapshot(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Check if operation is user-initiated (not automatic snapshot)
     * @returns {boolean}
     */
    is_user_initiated() {
        const ret = wasm.jjoperation_is_user_initiated(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Check if operation modifies history (for WASM)
     * @returns {boolean}
     */
    modifies_history() {
        const ret = wasm.jjoperation_modifies_history(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Check if operation is remote (for WASM)
     * @returns {boolean}
     */
    is_remote_operation() {
        const ret = wasm.jjoperation_is_remote_operation(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Get tags as JSON string (for WASM)
     * @returns {string}
     */
    get tags_json() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjoperation_tags_json(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Get metadata as JSON string (for WASM)
     * @returns {string}
     */
    get metadata_json() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjoperation_metadata_json(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
}
if (Symbol.dispose) JJOperation.prototype[Symbol.dispose] = JJOperation.prototype.free;

const JJResultFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_jjresult_free(ptr >>> 0, 1));
/**
 * Result wrapper for jujutsu operations
 *
 * This type extends the basic command result with metadata about execution,
 * warnings, and structured data.
 *
 * # Examples
 *
 * ```rust
 * use agentic_jujutsu::types::JJResult;
 *
 * let result = JJResult::new("output".to_string(), "".to_string(), 0, 100);
 * assert!(result.success());
 * ```
 */
export class JJResult {

    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        JJResultFinalization.unregister(this);
        return ptr;
    }

    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_jjresult_free(ptr, 0);
    }
    /**
     * Exit code
     * @returns {number}
     */
    get exit_code() {
        const ret = wasm.__wbg_get_jjconfig_max_log_entries(this.__wbg_ptr);
        return ret;
    }
    /**
     * Exit code
     * @param {number} arg0
     */
    set exit_code(arg0) {
        wasm.__wbg_set_jjconfig_max_log_entries(this.__wbg_ptr, arg0);
    }
    /**
     * Command execution time in milliseconds
     * @returns {bigint}
     */
    get execution_time_ms() {
        const ret = wasm.__wbg_get_jjconfig_timeout_ms(this.__wbg_ptr);
        return BigInt.asUintN(64, ret);
    }
    /**
     * Command execution time in milliseconds
     * @param {bigint} arg0
     */
    set execution_time_ms(arg0) {
        wasm.__wbg_set_jjconfig_timeout_ms(this.__wbg_ptr, arg0);
    }
    /**
     * Create a new JJResult
     * @param {string} stdout
     * @param {string} stderr
     * @param {number} exit_code
     * @param {bigint} execution_time_ms
     */
    constructor(stdout, stderr, exit_code, execution_time_ms) {
        const ptr0 = passStringToWasm0(stdout, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
        const len0 = WASM_VECTOR_LEN;
        const ptr1 = passStringToWasm0(stderr, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
        const len1 = WASM_VECTOR_LEN;
        const ret = wasm.jjresult_new(ptr0, len0, ptr1, len1, exit_code, execution_time_ms);
        this.__wbg_ptr = ret >>> 0;
        JJResultFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Get stdout (for WASM)
     * @returns {string}
     */
    get stdout() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjresult_stdout(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Get stderr (for WASM)
     * @returns {string}
     */
    get stderr() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjresult_stderr(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Check if the command was successful
     * @returns {boolean}
     */
    success() {
        const ret = wasm.jjresult_success(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Get the output as a string (prefer stdout, fallback to stderr)
     * @returns {string}
     */
    output() {
        let deferred1_0;
        let deferred1_1;
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.jjresult_output(retptr, this.__wbg_ptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            deferred1_0 = r0;
            deferred1_1 = r1;
            return getStringFromWasm0(r0, r1);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
            wasm.__wbindgen_export(deferred1_0, deferred1_1, 1);
        }
    }
}
if (Symbol.dispose) JJResult.prototype[Symbol.dispose] = JJResult.prototype.free;

export function __wbg___wbindgen_throw_b855445ff6a94295(arg0, arg1) {
    throw new Error(getStringFromWasm0(arg0, arg1));
};

export function __wbg_error_7534b8e9a36f1ab4(arg0, arg1) {
    let deferred0_0;
    let deferred0_1;
    try {
        deferred0_0 = arg0;
        deferred0_1 = arg1;
        console.error(getStringFromWasm0(arg0, arg1));
    } finally {
        wasm.__wbindgen_export(deferred0_0, deferred0_1, 1);
    }
};

export function __wbg_getRandomValues_38a1ff1ea09f6cc7() { return handleError(function (arg0, arg1) {
    globalThis.crypto.getRandomValues(getArrayU8FromWasm0(arg0, arg1));
}, arguments) };

export function __wbg_log_8cec76766b8c0e33(arg0) {
    console.log(getObject(arg0));
};

export function __wbg_new_8a6f238a6ece86ea() {
    const ret = new Error();
    return addHeapObject(ret);
};

export function __wbg_stack_0ed75d68575b0f3c(arg0, arg1) {
    const ret = getObject(arg1).stack;
    const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_export3, wasm.__wbindgen_export4);
    const len1 = WASM_VECTOR_LEN;
    getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
    getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
};

export function __wbindgen_cast_2241b6af4c4b2941(arg0, arg1) {
    // Cast intrinsic for `Ref(String) -> Externref`.
    const ret = getStringFromWasm0(arg0, arg1);
    return addHeapObject(ret);
};

export function __wbindgen_object_drop_ref(arg0) {
    takeObject(arg0);
};

