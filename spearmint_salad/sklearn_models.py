# -*- coding: utf-8 -*-
'''
Created on Oct 27, 2013

@author: alexandre
'''



import hp


# regression
#----------------
def gbr_space():
    from sklearn.ensemble import GradientBoostingRegressor
    
    return hp.Obj(GradientBoostingRegressor)(
        max_depth = hp.Int(1, 15 ),
        learning_rate = hp.Float( 0.01, 1, hp.log_scale ),
        n_estimators = 100,
    )

def gbr_space_3D():

    from sklearn.ensemble import GradientBoostingRegressor
    
    return hp.Obj(GradientBoostingRegressor)(
        max_depth = hp.Int(1, 15 ),
        learning_rate = hp.Float( 0.01, 1, hp.log_scale ),
        alpha = hp.Float( 0.0001, 1 ),
        loss = 'huber',
        n_estimators = 100,
    )



class Forest:

    forest_map = {
        'RFC':"RandomForestClassifier",
        'eTreeC':"ExtraTreesClassifier",
        'RFR':"RandomForestRegressor",
        'eTreeR':"ExtraTreesRegressor",
        }

    def __init__(self,algo='RFC', return_proba=False, base_prob=0.01, **argD):
        self.argD = argD
        self.algo= Forest.forest_map.get(algo, algo)
        self.return_proba = return_proba
        self.base_prob = base_prob

    def replace_max_features( self, n ):
        max_features = self.argD.get( "max_features","auto" )
        if isinstance(max_features, float):
            if max_features > 0 and max_features <= 1:
                max_features_new = int(round(n*max_features) ) 
                self.argD['max_features'] = max( max_features_new, 1 )
#                print 'replacing max features : %.3g*%d = %.3g'%(max_features, n, max_features_new)
    
    def fit(self,x,y):
        import sklearn.ensemble as ensemble
        learner = getattr( ensemble, self.algo ) # get the appropriate learner 
        self.replace_max_features( x.shape[1] ) # set max_features to the appropriate value
        self.predictor  = learner(**self.argD).fit(x,y)
        
        return self
        
    def predict( self, x):
        if self.return_proba:
            p =  self.predictor.predict_proba(x)
            p += self.base_prob
            p /= p.sum(1).reshape(-1,1)
            return p
            
        else:
            return self.predictor.predict(x)
        
def forest_regressor_space_3D():

    return hp.Obj(Forest)(
        n_estimators = 100,
        max_features = hp.Float(0.0001,1),
        bootstrap = hp.Enum(True, False),
        algo = hp.Enum('RFR','eTreeR')
    ) 
        


def svr_space_3D():

    from sklearn.svm import SVR
    
    return hp.Obj(SVR)(
        C = hp.Float( 0.01, 1000, hp.log_scale ),
        gamma = hp.Float( 10**-5, 1000, hp.log_scale ),
        epsilon = hp.Float(0.01,1, hp.log_scale),
    )
    
def svr_space_2D():

    from sklearn.svm import SVR
    return hp.Obj(SVR)(
        C = hp.Float( 0.01, 1000, hp.log_scale ),
        gamma = hp.Float( 10**-5, 1000, hp.log_scale ),
    )

# classification
#----------------

def svc_space():

    from sklearn.svm import SVC
    
    return hp.Obj(SVC)(
        C = hp.Float( 0.01, 1000, hp.log_scale ),
        gamma = hp.Float( 10**-5, 1000, hp.log_scale ),
    )
    
def svc_space_1d():

    from sklearn.svm import SVC
    
    return hp.Obj(SVC)(
#         C = hp.Float( 0.01, 1000, hp.log_scale ),
        C = 1000,
        gamma = hp.Float( 10**-5, 1000, hp.log_scale ),
    )

def forest_classifier_space_3D(n_estimators = 100):

    return hp.Obj(Forest)(
        n_estimators = n_estimators,
        max_features = hp.Float(0.0001,1),
        bootstrap = hp.Enum(True, False),
        algo = hp.Enum('RFC','eTreeC')
    )

def forest_nll_space_4D(n_estimators = 100):

    return hp.Obj(Forest)(
        n_estimators = n_estimators,
        base_prob = hp.Float(1e-8,1e-1, hp.log_scale),
        max_features = hp.Float(0.0001,1),
        bootstrap = hp.Enum(True, False),
        algo = hp.Enum('RFC','eTreeC'),
        return_proba = True,
    )

class GBC:


    def __init__(self,**argD):
        self.argD = argD

    def replace_max_features( self, n ):
        max_features = self.argD.get( "max_features","auto" )
        if isinstance(max_features, float):
            if max_features > 0 and max_features <= 1:
                max_features_new = int(round(n*max_features) ) 
                self.argD['max_features'] = max( max_features_new, 1 )
#                print 'replacing max features : %.3g*%d = %.3g'%(max_features, n, max_features_new)
    
    def fit(self,x,y):
        from sklearn.ensemble import GradientBoostingClassifier
        self.replace_max_features( x.shape[1] ) # set max_features to the appropriate value
        self.learner = GradientBoostingClassifier(**self.argD).fit(x,y)
        return self
    
    def predict(self,x):
        return self.learner.predict(x)


def gbc_space_3D():

    return hp.Obj(GBC)(
        max_depth = hp.Int(1, 15 ),
        learning_rate = hp.Float( 0.01, 1, hp.log_scale ),
        max_features = hp.Float( 0.001, 1 ),
        n_estimators = 100,
    )
    

        

