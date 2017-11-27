
const express = require("express");
const bodyParser = require("body-parser");
const app = express();
const static = express.static(__dirname + '/public');

const configRoutes = require("./routes");

const exphbs = require('express-handlebars');

const Handlebars = require('handlebars');

var passport = require('passport');
var LocalStrategy = require('passport-local').Strategy;

var cookieParser = require('cookie-parser');
var session = require('express-session');
var flash = require('express-flash');

const data = require("./data");
const user = data.user;
const admin = data.admin;

const handlebarsInstance = exphbs.create({
    defaultLayout: 'main',
    // Specify helpers which are only registered on this instance.
    helpers: {
        asJSON: (obj, spacing) => {
            console.log(obj.data.root.modelData);
            return JSON.stringify(obj.data.root.modelData);
            // if (typeof spacing === "number")
            //     return new Handlebars.SafeString(JSON.stringify(obj, null, spacing));
        
            // return new Handlebars.SafeString(JSON.stringify(obj));
        }
    },
     partialsDir: [
        'views/partials/'
    ]
});

const rewriteUnsupportedBrowserMethods = (req, res, next) => {
    // If the user posts to the server with a property called _method, rewrite the request's method
    // To be that method; so if they post _method=PUT you can now allow browsers to POST to a route that gets
    // rewritten in this middleware to a PUT route
    if (req.body && req.body._method) {
        req.method = req.body._method;
        delete req.body._method;
    }

    // let the next middleware run:
    next();
};

app.use("/public", static);
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(rewriteUnsupportedBrowserMethods);
app.use(cookieParser());

app.use(session({secret: 'huangzijing', cookie: { maxAge: 86400000 },resave:false,saveUninitialized:true}));
app.use(passport.initialize());
app.use(passport.session());
app.use(flash());


passport.use('localadmin', new LocalStrategy(
    function (adminname, password, done) {
        admin.checkAdminLogin(adminname,password).then((admin)=>{
            return done(null, admin);
        }).catch((err)=>{
            //console.log(err);
            return done(null, false, { message: err });
        });
    }
));

passport.serializeUser(function (user, done) {
    done(null, user);
});

passport.deserializeUser(function (user, done) {
    done(null, user);
});

app.engine('handlebars', handlebarsInstance.engine);
app.set('view engine', 'handlebars');

configRoutes(app);

app.listen(3001, () => {
    console.log("We've now got a server!");
    console.log("Your routes will be running on http://localhost:3001");
});
