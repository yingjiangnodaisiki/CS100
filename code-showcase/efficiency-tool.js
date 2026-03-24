class TaskStore {
  constructor() {
    this.tasks = [];
    this.nextId = 1;
  }

  add(title, priority = "medium") {
    const task = {
      id: this.nextId++,
      title,
      priority,
      done: false,
      createdAt: new Date().toISOString(),
    };
    this.tasks.push(task);
    return task;
  }

  toggle(id) {
    const task = this.tasks.find((item) => item.id === id);
    if (!task) return null;
    task.done = !task.done;
    return task;
  }

  list({ onlyPending = false } = {}) {
    return onlyPending ? this.tasks.filter((item) => !item.done) : this.tasks;
  }
}

const store = new TaskStore();
store.add("整理项目链接", "high");
store.add("完善 README", "medium");
store.toggle(1);
console.log(store.list());

