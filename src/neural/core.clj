(ns neural.core
  (:gen-class))

(use 'clojure.core.matrix)

(require '[clojure.data.csv :as csv] '[clojure.java.io :as io])

(defn activate 
"Sigmoid activation function"
[a]
  (into [] (map (fn [x] (/ 1. (+ 1 (Math/exp (- 0 x))))) a)))


(def layer-sizes [20,30,10])

(def weight-shapes (into [] (zipmap (rest layer-sizes) (drop-last layer-sizes))))


(defn randmat "Creates a random matrix of shape sh" [sh]  (reshape (repeatedly (* (sh 0) (sh 1)) #(rand)) sh))

(def weights  (into [] (for [ashape weight-shapes] (randmat ashape)))  )

(def biases (into [] (for [size (rest layer-sizes)] (vec (repeat size 0))))
)

(defn propagate1
"use weights and bias for one layer"
[x [w b]]
  ( activate (add (mmul w x) b)))

(defn propagate "propagate wieghts and biases from the input layer to the output layer"
[x w b]
(reduce propagate1 x (into [] (zipmap w b)))
)

(defn -main
  "main"
  [& args]
   (propagate [1 3] weights biases))
