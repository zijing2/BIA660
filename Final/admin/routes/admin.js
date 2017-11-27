var passport = require('passport');
const express = require('express');
const router = express.Router();
//var request = require('request')
var request = require('sync-request');

router.get("/login", (req, res) => {
        var flash = req.flash();
        var error = flash.error;
        var data = {
                "error": error,
                "layout": ""
        };
        res.render('adminlogin', data);
});

router.post("/login", passport.authenticate('localadmin', {
        successRedirect: '/admin',
        failureRedirect: '/admin/login',
        failureFlash: true
}));

router.all('/*', isLoggedIn);

router.get("/", (req, res) => {
        var data = {
                "layout": "",
                "is_predict": 1
        }
        res.render('admin', data);
});

router.get("/predict", (req, res) => {
        var data = {
                "layout": "",
                "is_predict": 1
        };

        res.render('admin', data);
});

router.get("/perform", (req, res) => {
        
        var rtn = request('GET', 'http://127.0.0.1:8887/reviewAnalyser/api/v1.0/performace/label');
        var label_perform = rtn.getBody().toString()
        var rtn = request('GET', 'http://127.0.0.1:8887/reviewAnalyser/api/v1.0/performace/sent');
        var sent_perform = rtn.getBody().toString()

        var data = {
                "layout": "",
                "is_perform": 1,
                "label_perform": label_perform,
                "sent_perform": sent_perform
        };
        res.render('admin', data);
});

router.get("/docinform", (req, res) => {
        
        var data = {
                "layout": "",
                "is_docinform": 1
        };

        res.render('admin', data);
});

router.post("/predict", (req, res) => {
        reviews = req.body["reviews"];

        var rtn = request('POST', 'http://127.0.0.1:8887/reviewAnalyser/api/v1.0/predict/review', {
                json: { reviews: reviews }
        });
        res.status(200).json(rtn.getBody().toString());
});  

router.get('/logout', function (req, res) {
        req.logout();
        res.redirect('/admin/login');
});

function isLoggedIn(req, res, next) {
        //console.log(req.isAuthenticated());
        //console.log(req.user);
        if (req.isAuthenticated() && req.user.adminname) {
                return next()
        }
        res.redirect('/admin/login');
}

module.exports = router;