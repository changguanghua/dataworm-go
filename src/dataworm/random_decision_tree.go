package dataworm


type TreeNode struct{
	left, right, depth int
	prediction float64
	samples []int
}

func (t *TreeNode) AddSample(k int) {
	t.samples = append(t.samples, k)
}

type Tree struct {
	nodes []TreeNode
}

func (t *Tree) AddTreeNode(n TreeNode){
	t.nodes = append(t.nodes, n)
}

type FeatureSplit struct {
	feature_id int64
	split_value float64
}

func GoLeft(sample MapBasedSample, feature_split FeatureSplit) bool {
	value, ok := sample.Features[feature_split.feature_id]
	if ok && value > feature_split.split_value {
		return false
	} else {
		return true
	}
}

func SingleRandomDTBuild(dataset DataSet, feature_splits []FeatureSplit) Tree {
	tree := Tree{}
	queue := make(chan TreeNode)
	root := TreeNode{depth: 0, left: -1, right: -1, prediction : -1, samples: []int{}}
	for i, _ := range dataset{
		root.AddSample(i)
	}
	queue <- root
	tree = append(tree, root)
	for {
		node := <-queue
		left_node := TreeNode{depth: node.depth + 1, left: -1, right: -1, prediction: -1, samples: []int{}}
		right_node := TreeNode{depth: node.depth + 1, left: -1, right: -1, prediction: -1, sample: []int{}}
		positive := 0.0
		total := 0.0
		for i, k := range node.samples {
			positive += dataset.Samples[k].LabelDoubleValue()
			total += 1.0
			if GoLeft(dataset.Samples[k], feature_splits[node.depth])	{
				left_node.samples = append(left_node.samples, k)	
			} else{
				right_node.samples = append(right_node.samples, k)
			}
		}
		
		node.prediction = positive / total
		node.samples = nil
		
		if len(left_node.samples) > 10{
			queue <- left_node
			node.left = len(tree)
			tree = append(tree, left_node)
		}
		
		if len(right_node.samples) > 10{
			queue <- right_node
			node.right = len(tree)
			tree = append(tree, right_node)
		}
	}
	
	return tree
}