(ns neural.core
  (:gen-class))

(use 'clojure.core.matrix)

(require '[clojure.data.csv :as csv] '[clojure.java.io :as io])

(defn read-row [reader row-index]
  (let [data (csv/read-csv reader)]
    (nth data row-index)))


(defn activate 
"Sigmoid activation function"
[a]
  (into [] (map (fn [x] (/ 1. (+ 1 (Math/exp (- 0 x))))) a)))

(defn sigmoid-prime [a] (let [b (activate a)] (mul b (sub 1 b))))

(def layer-sizes [2,3,4])

(def weight-shapes (into [] (zipmap (rest layer-sizes) (drop-last layer-sizes))))


(defn randmat "Creates a random matrix of shape sh" [sh]  (reshape (repeatedly (* (sh 0) (sh 1)) #(/ (rand)  (sh 1))) sh))


(def weights  (into [] (for [ashape weight-shapes] (randmat ashape)))  )

(def biases (into [] (for [size (rest layer-sizes)] (vec (repeat size 0))))
)

(defn propagate1
"use weights and bias for one layer"
[x [w b]]
  ( activate (add (mmul w x) b)))

(defn propagate "propagate weights and biases from the input layer to the output layer"
[x w b]
(reduce propagate1 x (into [] (zipmap w b)))
)

(defn accuracy [predictions label] (== label (.indexOf predictions (apply max predictions)))
)

(defn matsq "square matrix multiplication "[x] (mmul x x)) 

(defn cost [a y] (/ (matsq (sub a y)) (* 2 (count a))))

(defn dcost "cost derivative with respect to output" 
[a y] (sub a y)
)

(defn delta [z y] (let [a (activate z)] (mul (dcost a y) (mul (sub 1 a) a)) ))

(defn propadd
"use weights and bias for one layer and add to the list of layers"
[layers [w b]]
  (into  layers [ (add (mmul w (last layers)) b)])
)

(defn deltal [w z delta]
  (mul (mmul (transpose w) delta) (sigmoid-prime z))
)

(defn backprop [x w b]
  (let [layers  (reduce propadd [x] (into [] (zipmap w b))) outp [4 5 6 7]
        ]
    (loop [layer (- (count layer-sizes) 2) result (conj '() (delta (last layers) outp))]
      (if (< layer 0)
         result
        (recur (dec layer)  (conj result 
                                        (deltal (weights layer) (layers layer) (first result)) 
                      
 ) )) )
))

(defn -main
  "main"
  [& args]
   (propagate args))
