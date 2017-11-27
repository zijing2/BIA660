const adminRoutes = require("./admin");

const constructorMethod = (app) => {
    app.use("/admin", adminRoutes);
    app.use("*", adminRoutes);
};

module.exports = constructorMethod;